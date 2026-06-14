//! `QueryStatsAccumulator` — per-(layer, kv_head) Q(query) running mean/var (Welford online).
//!
//! Expected Attention(arXiv 2510.00636)이 query 분포의 running mean/var 로
//! *미래* attention 을 closed-form 추정하는 입력을 공급한다. `TensorKind::QueryStats` 소비자
//! (Expected Attention / 항목 4 read-plan 의 Quest 류 page 선택)가 kv_head 마다 `(μ, σ²)` 쌍을
//! 읽는다.
//!
//! **설계 격리(U4)**: `AttentionScoreAccumulator` 패턴을 *참조*하되 **별도 struct·별도 모듈**로
//! 격리한다 — `attention_scores.rs`/`qcf_runtime.rs` 의 기존 누적 로직은 무수정(QCF_kv 설계 라운드
//! 충돌 방지). QueryStats 는 전부 신규 코드.
//!
//! ## 좌표계 (MQ-1/MQ-2)
//!
//! - per-(layer, kv_head, dim) Welford `(count, mean, M2)` triple 을 유지한다.
//! - GQA 환원: `accumulate_layer` 가 Q-head(예 32) → kv_head(예 8) 그룹 내 element-wise 평균
//!   (`AttentionScoreAccumulator::accumulate_layer_gqa` 의 `inv_rep` 패턴과 동형)으로 환원한 1 sample
//!   을 각 (layer, kv_head, dim) Welford 누적기에 추가한다. 이로써 QueryStats 와 Scores 가 같은
//!   kv_head 좌표를 공유(소비자 교차 사용 가능).
//! - `layer_stats(layer)` = `[n_kv_heads * 2 * head_dim]` 평탄 슬라이스 (row0=mean / row1=var,
//!   `KVStageCtx` 가 단일-layer ctx 에 바인딩, MQ-4 (c)).
//!
//! ## 수명 (MQ-3/MQ-4)
//!
//! score accumulator 와 동형(`ScoreCell` 패턴): decode step 마다 활성 layer 의 **RoPE-적용 Q**(`ws.q`)
//! 를 1 sample 누적, eviction 시 `reset`. **prefill 미누적**(MQ-4 게이트 — score-active 폴백 경로에서만
//! 진입, prefill 은 `args.workspace=None` 이라 누적 seam 미진입). 기본 off(`active=false`).

/// per-(layer, kv_head) Q running mean/var 누적기 (Welford online, MQ-3).
///
/// 상태: per-(layer, kv_head, dim) `(count, mean, M2)`. `var = M2/count`. decode step 마다
/// 활성 layer 의 RoPE-적용 Q 를 GQA 환원해 1 sample 추가.
pub struct QueryStatsAccumulator {
    /// 디코더 layer 수.
    n_layers: usize,
    /// KV head 수 (GQA 환원 대상 좌표).
    n_kv_heads: usize,
    /// head 당 차원.
    head_dim: usize,
    /// Q-head 수 (GQA 환원 입력, `n_heads_q / n_kv_heads = n_rep`).
    n_heads_q: usize,
    /// per-(layer, kv_head, dim) Welford count. `[n_layers * n_kv_heads]` (dim 별 동일 count라
    /// per-(layer,kv_head) 1개로 충분 — 한 sample 이 head_dim 전체를 동시 갱신).
    count: Vec<u64>,
    /// per-(layer, kv_head, dim) running mean. `[n_layers * n_kv_heads * head_dim]`, row-major.
    mean: Vec<f32>,
    /// per-(layer, kv_head, dim) running M2 (sum of squared deviations). 같은 레이아웃.
    m2: Vec<f32>,
    /// 누적 활성 여부 (기본 off — score-active 시만 true, MQ-4 hot-path 게이트).
    active: bool,
    /// `layer_stats` 출력 스크래치 `[n_kv_heads * 2 * head_dim]` (row0=mean/row1=var). 호출 시 재계산.
    stats_scratch: Vec<f32>,
}

impl QueryStatsAccumulator {
    /// `(layer, kv_head, dim)` 좌표를 만든다. GQA 환원: Q-head 수 `n_heads_q` 는 환원 입력.
    pub fn new(n_layers: usize, n_heads_q: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        debug_assert!(n_kv_heads > 0, "n_kv_heads must be > 0");
        debug_assert!(
            n_heads_q.is_multiple_of(n_kv_heads),
            "n_heads_q must be divisible by n_kv_heads (GQA)"
        );
        let lh = n_layers * n_kv_heads;
        let lhd = lh * head_dim;
        Self {
            n_layers,
            n_kv_heads,
            head_dim,
            n_heads_q,
            count: vec![0; lh],
            mean: vec![0.0; lhd],
            m2: vec![0.0; lhd],
            active: false,
            stats_scratch: vec![0.0; n_kv_heads * 2 * head_dim],
        }
    }

    /// 누적 활성화/비활성화 (score-active 게이트와 동형, MQ-4).
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// 활성 여부.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// KV head 수.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// head 당 차원.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// 한 layer 의 RoPE-적용 Q `[n_heads_q * head_dim]`(decode seq_len=1)에서 GQA 환원 1 sample 을
    /// 그 layer 의 per-kv_head Welford 누적기에 추가한다(MQ-2/MQ-3).
    ///
    /// `q`: 한 layer 의 Q 버퍼 — `ws.q.as_slice::<f32>()`. 최소 `n_heads_q * head_dim` 길이여야 한다.
    /// 비활성(`active=false`) 또는 `layer >= n_layers` 면 no-op.
    ///
    /// GQA 환원: kv_head `h` 의 sample[d] = 그룹 `n_rep` 개 Q-head 의 `q[(h*n_rep + r)*head_dim + d]`
    /// element-wise 평균(`accumulate_layer_gqa` 의 `inv_rep` 패턴). 그 sample 을 Welford 1-sample
    /// 갱신(`count += 1`, `delta = x - mean`, `mean += delta/count`, `M2 += delta*(x - mean)`)으로
    /// 누적한다.
    pub fn accumulate_layer(&mut self, q: &[f32], layer: usize) {
        if !self.active || layer >= self.n_layers {
            return;
        }
        let n_rep = self.n_heads_q / self.n_kv_heads;
        let inv_rep = 1.0 / n_rep as f32;
        let hd = self.head_dim;

        for kv_h in 0..self.n_kv_heads {
            let lh_idx = layer * self.n_kv_heads + kv_h;
            // Welford count 갱신 (head_dim 전체가 1 sample 동시 진입).
            let new_count = self.count[lh_idx] + 1;
            self.count[lh_idx] = new_count;
            let inv_count = 1.0 / new_count as f32;
            let base = lh_idx * hd;

            for d in 0..hd {
                // GQA 환원: 그룹 내 Q-head element-wise 평균 → sample x.
                let mut x = 0.0f32;
                for r in 0..n_rep {
                    x += q[(kv_h * n_rep + r) * hd + d];
                }
                x *= inv_rep;

                // Welford 1-sample online 갱신.
                let delta = x - self.mean[base + d];
                self.mean[base + d] += delta * inv_count;
                let delta2 = x - self.mean[base + d];
                self.m2[base + d] += delta * delta2;
            }
        }
    }

    /// 한 layer 의 per-kv_head Q 통계를 `[n_kv_heads * 2 * head_dim]` 평탄 슬라이스로 산출한다(MQ-1).
    ///
    /// 레이아웃: `out[kv_head * 2 * head_dim + stat_row * head_dim + d]`,
    /// `stat_row 0 = mean / 1 = var`(`var = M2/count`, count<2 면 0). `QueryStatsHandle` 이 이
    /// 슬라이스 위에서 `read_row(row, kv_head, out)` 를 구현한다.
    ///
    /// `layer >= n_layers` 면 빈 슬라이스. 내부 스크래치를 매 호출 재계산(borrow 충돌 회피용
    /// `&mut self`).
    pub fn layer_stats(&mut self, layer: usize) -> &[f32] {
        if layer >= self.n_layers {
            return &[];
        }
        let hd = self.head_dim;
        for kv_h in 0..self.n_kv_heads {
            let lh_idx = layer * self.n_kv_heads + kv_h;
            let src_base = lh_idx * hd;
            let count = self.count[lh_idx];
            let out_base = kv_h * 2 * hd;
            for d in 0..hd {
                // row0 = mean.
                self.stats_scratch[out_base + d] = self.mean[src_base + d];
                // row1 = var = M2 / count (population variance; count<2 → 0).
                self.stats_scratch[out_base + hd + d] = if count >= 2 {
                    self.m2[src_base + d] / count as f32
                } else {
                    0.0
                };
            }
        }
        &self.stats_scratch
    }

    /// 전 누적 상태 초기화 (eviction 후, MQ-3). count/mean/M2 전부 0.
    pub fn reset(&mut self) {
        self.count.fill(0);
        self.mean.fill(0.0);
        self.m2.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TQS-1: Welford 단일 sample → mean=q, var=0 (count=1).
    #[test]
    fn tqs1_single_sample_mean_is_q_var_zero() {
        // n_layers=1, n_heads_q=2, n_kv_heads=2, head_dim=3 (n_rep=1 → 환원 = identity).
        let mut acc = QueryStatsAccumulator::new(1, 2, 2, 3);
        acc.set_active(true);
        // q = [kv0: 1,2,3] [kv1: 4,5,6].
        acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0);
        let s = acc.layer_stats(0);
        // kv0 mean = [1,2,3], var = [0,0,0] (count=1 → 0).
        assert_eq!(&s[0..3], &[1.0, 2.0, 3.0], "kv0 mean = q");
        assert_eq!(&s[3..6], &[0.0, 0.0, 0.0], "kv0 var = 0 (count<2)");
        // kv1 mean = [4,5,6], var = [0,0,0].
        assert_eq!(&s[6..9], &[4.0, 5.0, 6.0], "kv1 mean = q");
        assert_eq!(&s[9..12], &[0.0, 0.0, 0.0], "kv1 var = 0");
    }

    /// TQS-2: Welford N sample mean/var = 2-pass ground-truth `<1e-5`.
    #[test]
    fn tqs2_welford_matches_two_pass() {
        // n_layers=1, n_heads_q=1, n_kv_heads=1, head_dim=1 (스칼라 분포 정밀 비교).
        let mut acc = QueryStatsAccumulator::new(1, 1, 1, 1);
        acc.set_active(true);
        let samples = [2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        for &x in &samples {
            acc.accumulate_layer(&[x], 0);
        }
        let s = acc.layer_stats(0);
        let got_mean = s[0];
        let got_var = s[1];
        // 2-pass ground-truth.
        let n = samples.len() as f32;
        let mean_ref = samples.iter().sum::<f32>() / n;
        let var_ref = samples.iter().map(|x| (x - mean_ref).powi(2)).sum::<f32>() / n;
        assert!(
            (got_mean - mean_ref).abs() < 1e-5,
            "mean {got_mean} vs ref {mean_ref}"
        );
        assert!(
            (got_var - var_ref).abs() < 1e-5,
            "var {got_var} vs ref {var_ref}"
        );
    }

    /// TQS-3: GQA 환원(n_heads_q=4, n_kv_heads=2): Q 0/1→kv0, 2/3→kv1 element-wise 평균.
    #[test]
    fn tqs3_gqa_reduction_averages_within_group() {
        // n_rep = 4/2 = 2. head_dim=2.
        let mut acc = QueryStatsAccumulator::new(1, 4, 2, 2);
        acc.set_active(true);
        // Q-head layout (head_dim=2):
        //   Q0=[1,2] Q1=[3,4] → kv0 = avg = [2,3]
        //   Q2=[10,20] Q3=[30,40] → kv1 = avg = [20,30]
        acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], 0);
        let s = acc.layer_stats(0);
        // kv0 mean = [2,3], var = 0 (single sample).
        assert_eq!(&s[0..2], &[2.0, 3.0], "kv0 = avg(Q0,Q1)");
        // kv1 mean (out_base = 1*2*2 = 4): [20,30].
        assert_eq!(&s[4..6], &[20.0, 30.0], "kv1 = avg(Q2,Q3)");
    }

    /// TQS-4: layer 격리 — layer 0 누적이 layer 1 통계 누설 0.
    #[test]
    fn tqs4_layer_isolation() {
        // n_layers=2, n_heads_q=1, n_kv_heads=1, head_dim=2.
        let mut acc = QueryStatsAccumulator::new(2, 1, 1, 2);
        acc.set_active(true);
        acc.accumulate_layer(&[5.0, 7.0], 0);
        // layer 1 은 미누적 → mean/var 전부 0.
        let s1 = acc.layer_stats(1);
        assert_eq!(&s1[0..2], &[0.0, 0.0], "layer 1 mean 누설 0");
        assert_eq!(&s1[2..4], &[0.0, 0.0], "layer 1 var 누설 0");
        // layer 0 은 정상.
        let s0 = acc.layer_stats(0);
        assert_eq!(&s0[0..2], &[5.0, 7.0], "layer 0 mean");
    }

    /// TQS-5: reset → count/mean/m2 전부 0.
    #[test]
    fn tqs5_reset_zeros_all() {
        let mut acc = QueryStatsAccumulator::new(1, 2, 2, 2);
        acc.set_active(true);
        acc.accumulate_layer(&[1.0, 2.0, 3.0, 4.0], 0);
        acc.accumulate_layer(&[5.0, 6.0, 7.0, 8.0], 0);
        acc.reset();
        let s = acc.layer_stats(0);
        assert!(s.iter().all(|&v| v == 0.0), "reset 후 전부 0");
        // count 도 0 → 추가 sample 이 다시 count=1 부터 시작(var=0).
        acc.accumulate_layer(&[9.0, 9.0, 9.0, 9.0], 0);
        let s2 = acc.layer_stats(0);
        assert_eq!(&s2[0..2], &[9.0, 9.0], "reset 후 첫 sample = mean");
        assert_eq!(
            &s2[2..4],
            &[0.0, 0.0],
            "reset 후 첫 sample = var 0 (count=1)"
        );
    }

    /// TQS-6: off(`set_active(false)`) 무회귀 — 누적 진입 없음.
    #[test]
    fn tqs6_inactive_no_accumulation() {
        let mut acc = QueryStatsAccumulator::new(1, 1, 1, 2);
        // active defaults to false.
        assert!(!acc.is_active());
        acc.accumulate_layer(&[100.0, 200.0], 0);
        let s = acc.layer_stats(0);
        assert!(s.iter().all(|&v| v == 0.0), "inactive → 누적 0");
    }

    /// 추가: layer 범위 밖 accumulate/layer_stats 는 안전 (no-op / 빈 슬라이스).
    #[test]
    fn out_of_range_layer_is_safe() {
        let mut acc = QueryStatsAccumulator::new(1, 1, 1, 2);
        acc.set_active(true);
        acc.accumulate_layer(&[1.0, 2.0], 5); // layer 5 >= n_layers → no-op.
        assert!(acc.layer_stats(5).is_empty(), "범위 밖 layer → 빈 슬라이스");
        let s = acc.layer_stats(0);
        assert!(s.iter().all(|&v| v == 0.0), "layer 0 미오염");
    }
}
