//! head importance 분산 측정 — Ada-KV(arXiv 2407.11550) 전제 검증 (P2c).
//!
//! per-KV-head attention concentration C_h = 상위 5% 토큰이 차지하는 attention 질량.
//! GQA mode에서 `last_step_head_attn()` 출력 `[n_kv_heads * seq_len]`을 입력으로 받아
//! per-(layer, kv_head) C_h 행렬과 max/min ratio 요약을 산출한다.

/// 단일 KV-head의 softmax attention 분포에서 상위 `top_frac` 질량 비율(C_h)을 계산한다.
///
/// `attn`: 해당 head의 attention weights(`[seq_len]`, 합 ≈ 1.0).
/// `top_frac`: 상위 토큰 비율(예: 0.05 = 상위 5%).
/// 반환: 상위 `ceil(top_frac * seq_len)` 토큰의 attention 질량 합.
pub fn compute_ch(attn: &[f32], top_frac: f32) -> f32 {
    if attn.is_empty() {
        return 0.0;
    }
    // 내림차순 정렬 (원본 불변)
    let mut sorted = attn.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let top_k = ((attn.len() as f32 * top_frac).ceil() as usize).max(1);
    sorted[..top_k.min(sorted.len())].iter().sum()
}

/// 여러 step에서 누적한 per-(layer, kv_head) C_h 누적합과 step 카운터를 관리한다.
#[derive(Debug, Default)]
pub struct ConcentrationAccumulator {
    /// `[n_layers * n_kv_heads]` — 각 (layer, head)의 C_h 누적합.
    sum: Vec<f64>,
    /// 누적된 step 수.
    steps: u64,
    n_kv_heads: usize,
}

impl ConcentrationAccumulator {
    pub fn new(n_layers: usize, n_kv_heads: usize) -> Self {
        Self {
            sum: vec![0.0; n_layers * n_kv_heads],
            steps: 0,
            n_kv_heads,
        }
    }

    /// 단일 decode step에서 모든 layer의 `last_step_head_attn` 슬라이스를 누적한다.
    ///
    /// `layer_attns`: layer별 `[n_kv_heads * seq_len]` 슬라이스 목록.
    /// `seq_len`: 현재 유효 시퀀스 길이(head stride와 동일, softmax 범위).
    pub fn accumulate(&mut self, layer_attns: &[Option<Vec<f32>>], seq_len: usize, top_frac: f32) {
        for (layer_idx, maybe_attn) in layer_attns.iter().enumerate() {
            let Some(attn) = maybe_attn else { continue };
            for h in 0..self.n_kv_heads {
                let start = h * seq_len;
                let end = (start + seq_len).min(attn.len());
                if start >= attn.len() {
                    continue;
                }
                let ch = compute_ch(&attn[start..end], top_frac);
                let idx = layer_idx * self.n_kv_heads + h;
                if idx < self.sum.len() {
                    self.sum[idx] += ch as f64;
                }
            }
        }
        self.steps += 1;
    }

    /// `last_step_head_attn()` 버퍼(`[n_kv_heads * stride]` 레이아웃)를 위한 누적.
    ///
    /// `last_step_head_attn`의 레이아웃은 `[n_kv_heads * max_seq_len]`이므로
    /// head stride = max_seq_len, 유효 토큰 수 = valid_len (별도 파라미터).
    ///
    /// `stride`: 버퍼에서 각 head의 간격 (= max_seq_len).
    /// `valid_len`: 현재 decode step의 유효 시퀀스 길이.
    pub fn accumulate_strided(
        &mut self,
        layer_attns: &[Option<Vec<f32>>],
        stride: usize,
        valid_len: usize,
        top_frac: f32,
    ) {
        for (layer_idx, maybe_attn) in layer_attns.iter().enumerate() {
            let Some(attn) = maybe_attn else { continue };
            for h in 0..self.n_kv_heads {
                let start = h * stride;
                let end = (start + valid_len).min(attn.len());
                if start >= attn.len() {
                    continue;
                }
                let ch = compute_ch(&attn[start..end], top_frac);
                let idx = layer_idx * self.n_kv_heads + h;
                if idx < self.sum.len() {
                    self.sum[idx] += ch as f64;
                }
            }
        }
        self.steps += 1;
    }

    /// step 평균 C_h 행렬을 반환한다 (f32 벡터, layout `[n_layers * n_kv_heads]`).
    pub fn mean_ch(&self) -> Vec<f32> {
        if self.steps == 0 {
            return vec![0.0; self.sum.len()];
        }
        let steps = self.steps as f64;
        self.sum.iter().map(|&s| (s / steps) as f32).collect()
    }

    /// max C_h / min C_h 비율(스칼라)을 계산한다.
    /// Ada-KV 판정 임계: <2배 → 항목 6 보류, ≥5배 → 개봉 후보.
    pub fn max_min_ratio(mean: &[f32]) -> f32 {
        let valid: Vec<f32> = mean.iter().copied().filter(|&v| v.is_finite()).collect();
        if valid.is_empty() {
            return 1.0;
        }
        let max_val = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_val = valid
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .fold(f32::INFINITY, f32::min);
        if min_val == f32::INFINITY || min_val == 0.0 {
            return 1.0;
        }
        max_val / min_val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 한 토큰에 모든 질량이 집중 → C_h ≈ 1.0 (seq_len=1이면 top 5%=1 토큰).
    #[test]
    fn test_ch_fully_concentrated() {
        // seq_len=20, 하나에 1.0, 나머지 0.0 → 상위 5% = ceil(1) = 1 토큰 → C_h = 1.0
        let mut attn = vec![0.0f32; 20];
        attn[0] = 1.0;
        let ch = compute_ch(&attn, 0.05);
        assert!((ch - 1.0).abs() < 1e-6, "fully concentrated: C_h={ch}");
    }

    /// 완전 균등 분포 → C_h ≈ top_frac (상위 5% = 5% 질량).
    #[test]
    fn test_ch_uniform() {
        let n = 100usize;
        let attn = vec![1.0 / n as f32; n];
        let ch = compute_ch(&attn, 0.05);
        // 상위 5% = 5개 × (1/100) = 0.05
        assert!((ch - 0.05).abs() < 1e-5, "uniform: C_h={ch}");
    }

    /// 빈 슬라이스 → 0.0.
    #[test]
    fn test_ch_empty() {
        assert_eq!(compute_ch(&[], 0.05), 0.0);
    }

    /// max_min_ratio: 집중 head와 균등 head 혼재 시 비율 > 1.
    #[test]
    fn test_max_min_ratio_varied() {
        let mean = vec![0.9f32, 0.1f32, 0.5f32, 0.05f32];
        let ratio = ConcentrationAccumulator::max_min_ratio(&mean);
        assert!(ratio > 1.0, "ratio={ratio} should be >1 for varied C_h");
    }

    /// max_min_ratio: 모두 같은 값 → 1.0.
    #[test]
    fn test_max_min_ratio_uniform() {
        let mean = vec![0.3f32; 4];
        let ratio = ConcentrationAccumulator::max_min_ratio(&mean);
        assert!(
            (ratio - 1.0).abs() < 1e-5,
            "ratio={ratio} should be 1.0 for uniform"
        );
    }

    /// accumulate: 단일 layer, 2 head — 집중 vs 균등 → 다른 C_h.
    #[test]
    fn test_accumulator_one_step() {
        let n_layers = 1;
        let n_kv_heads = 2;
        let seq_len = 20;
        let mut acc = ConcentrationAccumulator::new(n_layers, n_kv_heads);

        // head0: 한 토큰에 집중, head1: 균등
        let mut attn = vec![0.0f32; n_kv_heads * seq_len];
        attn[0] = 1.0; // head0[0] = 1.0
        for i in 0..seq_len {
            attn[seq_len + i] = 1.0 / seq_len as f32; // head1: uniform
        }

        acc.accumulate(&[Some(attn)], seq_len, 0.05);
        let mean = acc.mean_ch();
        assert_eq!(mean.len(), n_kv_heads);
        // head0 C_h ≈ 1.0, head1 C_h ≈ 0.05
        assert!(mean[0] > 0.9, "concentrated head: C_h={}", mean[0]);
        assert!(mean[1] < 0.1, "uniform head: C_h={}", mean[1]);
    }

    /// max_min_ratio: 집중 head(C_h≈1.0) vs 균등 head(C_h≈0.05) → ratio ≈ 20.
    #[test]
    fn test_max_min_ratio_concentrated_vs_uniform() {
        let mean = vec![1.0f32, 0.05f32];
        let ratio = ConcentrationAccumulator::max_min_ratio(&mean);
        assert!(
            ratio >= 5.0,
            "ratio={ratio} should be >= 5 (개봉 후보 임계)"
        );
    }
}
