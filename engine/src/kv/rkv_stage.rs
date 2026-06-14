//! R-KV 측정 프로토타입 — cosine redundancy + importance joint eviction stage.
//!
//! arXiv 2505.24133 (NeurIPS'25). KV roadmap 항목 0 측정 스프린트(P2a):
//! `arch/kv_roadmap_item0_measurement.md` §2.1 / §4.1.
//!
//! **측정 전용 — feature `rkv` 게이트**(CAOTE 선례 동형). production 빌드(feature OFF)에서는
//! 이 모듈 전체가 미컴파일 → `eviction rkv` subcommand 부재 → production 정책 카탈로그 불변
//! (빌드 타임 격리, §6 Spec Triage). 측정 GO 시 production 승격은 별도 Triage.
//!
//! **수식** (per-KV-head, §2.1):
//! 1. redundancy R: K(key) pairwise cosine N×N 행렬의 row-mean → softmax 정규화.
//! 2. importance I: 기존 accumulator score 재사용(설계서 근사 허용), 최근 α window.
//! 3. fusion Z = λ·I − (1−λ)·R, **λ=0.1**(redundancy 지배).
//! 4. 최근 α=8 항상 보존 + 나머지 budget 을 Z 상위 single-shot top-k.
//!
//! **GQA**: redundancy/importance 모두 KV-head 단위(8개)로 측정. head 별 Z 를 평균해 단일
//! layer-wide keep 산출(per-head 차등 keep 은 §6 항목 6 영역 — 본 프로토타입은 layer-wide 근사).
//!
//! **재사용**(§4.1): K 읽기는 `StageCtx::dequant_k`(d2o_handler `dequantize_k` 정본 위임),
//! cosine 은 `d2o_handler::cosine_similarity`. N×N row-mean 집계 루프만 신규.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use technique_api::{KVCachePlan, KVCacheStage, KeepSpec, StageCtx, TensorKind};

use crate::kv::d2o_handler::cosine_similarity;

/// 1단 측정 덤프 게이트 env var. set 시 plan() 마다 per-kv_head `[RkvStats]` 마커 라인을 stderr 로
/// 출력한다(파싱 가능 포맷). 측정 전용 — 미설정 시 덤프 경로 미진입(production 무영향).
const RKV_DUMP_ENV: &str = "ARGUS_RKV_DUMP";

/// fusion 가중치 λ (Z = λ·I − (1−λ)·R). 기본 0.1 = redundancy 지배(논문 §2.1).
pub const RKV_DEFAULT_LAMBDA: f32 = 0.1;
/// redundant fraction 의 nearest-neighbour cosine 임계 τ (§5 표, D2O cosine 정합).
pub const RKV_TAU: f32 = 0.5;
/// 항상 보존하는 최근 토큰 수 α (importance window 겸용).
pub const RKV_RECENT_ALPHA: usize = 8;

/// R-KV 측정 전용 설정(CLI 노출 안 함 — 측정 schedule 내부 상수). λ 만 측정 조정 가능.
#[derive(Clone, Copy, Debug)]
pub struct RkvConfig {
    /// fusion 가중치 λ.
    pub lambda: f32,
    /// 항상 보존하는 최근 토큰 수 α.
    pub recent_alpha: usize,
    /// nearest-neighbour redundancy 임계 τ.
    pub tau: f32,
}

impl Default for RkvConfig {
    fn default() -> Self {
        Self {
            lambda: RKV_DEFAULT_LAMBDA,
            recent_alpha: RKV_RECENT_ALPHA,
            tau: RKV_TAU,
        }
    }
}

/// 1단 측정 게이트 지표(§2.1-C). per-(layer, kv_head) 로 덤프한다.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RedundancyStats {
    /// MPC = mean pairwise K-cosine (N×N 비대각 평균).
    pub mpc: f32,
    /// redundant fraction = nearest-neighbour cosine > τ 인 토큰 비율(0..1).
    pub redundant_fraction: f32,
}

/// R-KV stage — `KVCacheStage` 구현체. λ/α/τ 는 [`RkvConfig`] 보유(plan 간 불변, 상태는 없음).
///
/// `Mutex<RkvConfig>`는 불필요하나(불변), 측정 hook 이 마지막 계산 stats 를 보관하도록 한다 —
/// 측정 schedule 이 plan() 호출 후 [`last_stats`](Self::last_stats)로 읽는다(stderr/CSV 덤프).
pub struct RkvStage {
    config: RkvConfig,
    /// 마지막 plan() 의 per-kv_head redundancy stats (측정 1단 덤프용). plan 은 `&self` 라 내부가변.
    last_stats: Mutex<Vec<RedundancyStats>>,
    /// plan() 호출 순번 — `[RkvStats]` 덤프의 `layer=` 필드. eviction 은 layer cache 를 순차 호출
    /// (KVStageCtx.layer_idx() 는 0 고정)하므로 호출 카운터가 layer 진행을 누적 추적한다. 측정
    /// schedule 이 `% num_layers` 로 layer 인덱스 역산. 누적이므로 reset 안 함(측정 전용).
    plan_calls: AtomicUsize,
}

impl RkvStage {
    /// 기본 설정(λ=0.1, α=8, τ=0.5)으로 생성.
    pub fn new(config: RkvConfig) -> Self {
        Self {
            config,
            last_stats: Mutex::new(Vec::new()),
            plan_calls: AtomicUsize::new(0),
        }
    }

    /// 직전 plan() 이 계산한 per-kv_head redundancy stats 의 복사본(측정 1단 덤프 진입점).
    /// plan() 호출 전이면 빈 Vec.
    pub fn last_stats(&self) -> Vec<RedundancyStats> {
        self.last_stats.lock().expect("rkv stats poisoned").clone()
    }
}

/// per-kv_head redundancy stats 를 `[RkvStats]` 마커 라인으로 stderr 덤프(env `ARGUS_RKV_DUMP` 게이트).
/// 포맷(파싱 가능): `[RkvStats] layer=<L> head=<H> mpc=<X> fraction=<Y>`. `layer` = plan() 호출 순번
/// (측정 schedule 이 num_layers 로 modulo). 측정 전용 — feature `rkv` 안에 격리.
fn dump_redundancy_stats(layer: usize, stats: &[RedundancyStats]) {
    if std::env::var_os(RKV_DUMP_ENV).is_none() {
        return;
    }
    for (head, s) in stats.iter().enumerate() {
        eprintln!(
            "[RkvStats] layer={layer} head={head} mpc={:.6} fraction={:.6}",
            s.mpc, s.redundant_fraction
        );
    }
}

impl KVCacheStage for RkvStage {
    fn name(&self) -> &str {
        "rkv"
    }

    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let n = ctx.current_pos();
        let target = ctx.target_len();
        // budget 이 현 토큰 수 이상이면 evict 불요(no-op). 0 토큰도 no-op.
        if n == 0 || target >= n {
            return None;
        }
        // K 텐서가 없으면(score-free 엔진) redundancy 계산 불가 → no-op. 핸들은 dequant_k sugar 로
        // 재접근하므로 존재 여부만 게이트(`?` 로 early-return).
        ctx.tensor(TensorKind::Key)?;

        let n_kv_heads = ctx.n_kv_heads().max(1);
        let head_dim = ctx.head_dim();
        let importance = ctx.importance();

        // per-KV-head K 행렬을 dequant 해 N×N cosine row-mean → R 산출 후 head 평균.
        // Z[t] = λ·I[t] − (1−λ)·R[t] (head 평균). 최근 α 항상 보존 + 나머지 top-k(Z).
        let mut k_rows = vec![0.0f32; n * head_dim]; // 재사용 버퍼(per head)
        let mut z_sum = vec![0.0f32; n]; // head 합산 Z
        let mut stats = Vec::with_capacity(n_kv_heads);

        for h in 0..n_kv_heads {
            // (1) head h 의 모든 토큰 K 를 dequant.
            for (t, row) in k_rows.chunks_mut(head_dim).enumerate().take(n) {
                ctx.dequant_k(t, h, row);
            }
            // (2) N×N pairwise cosine → row-mean R + 1단 게이트 지표(MPC, redundant fraction).
            //     softmax 정규화(§2.1 R 정의) 후 fusion 에 사용.
            let (mut r, stat) = redundancy_row_mean(&k_rows, n, head_dim, self.config.tau);
            softmax_in_place(&mut r);
            // (3) importance I 를 최근 α window 로 추출(설계서 근사: accumulator score 재사용).
            //     I[t] = importance[t] (score-based) 또는 0(score-free).
            for t in 0..n {
                let i = importance
                    .and_then(|imp| imp.get(t).copied())
                    .unwrap_or(0.0);
                // Z = λ·I − (1−λ)·R. head 합산(후에 평균).
                z_sum[t] += self.config.lambda * i - (1.0 - self.config.lambda) * r[t];
            }
            stats.push(stat);
        }

        let inv_heads = 1.0 / n_kv_heads as f32;
        for z in z_sum.iter_mut() {
            *z *= inv_heads;
        }

        // 1단 측정 hook: per-kv_head redundancy stats 를 stderr 마커로 덤프(env 게이트) 후 보관.
        let layer = self.plan_calls.fetch_add(1, Ordering::Relaxed);
        dump_redundancy_stats(layer, &stats);
        *self.last_stats.lock().expect("rkv stats poisoned") = stats;

        let keep = select_keep(&z_sum, n, target, self.config.recent_alpha);
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep),
            merges: Vec::new(),
        })
    }
}

/// N×N pairwise cosine 행렬의 **row-mean redundancy R** + 1단 게이트 지표(MPC, redundant fraction).
///
/// `k_rows`: `[n * head_dim]` row-major (토큰 t 의 K 벡터 = `k_rows[t*head_dim..(t+1)*head_dim]`).
/// 반환 `(r, stats)`:
/// - `r[t]` = 토큰 t 의 다른 토큰들과의 cosine 평균 (대각 제외). N=1 이면 0.
/// - `stats.mpc` = 전체 비대각 cosine 평균(스칼라).
/// - `stats.redundant_fraction` = nearest-neighbour cosine > τ 인 토큰 비율.
///
/// 반환 `r` 은 **raw row-mean**(softmax 미적용) — §2.1 정의의 softmax 정규화는 호출부(stage plan)
/// 가 [`softmax_in_place`]로 적용해 fusion Z 에 쓴다(단위 테스트는 두 단계를 분리 검증). 1단 게이트
/// (MPC/redundant fraction)는 R 정규화와 무관한 원시 cosine 통계이므로 본 함수가 직접 산출한다.
pub(crate) fn redundancy_row_mean(
    k_rows: &[f32],
    n: usize,
    head_dim: usize,
    tau: f32,
) -> (Vec<f32>, RedundancyStats) {
    let mut r = vec![0.0f32; n];
    if n <= 1 {
        return (
            r,
            RedundancyStats {
                mpc: 0.0,
                redundant_fraction: 0.0,
            },
        );
    }

    let mut pair_sum = 0.0f64; // MPC 누적(비대각, 대칭이라 i<j 만)
    let mut pair_count = 0u64;
    let mut nn_max = vec![f32::NEG_INFINITY; n]; // nearest-neighbour cosine

    for i in 0..n {
        let a = &k_rows[i * head_dim..(i + 1) * head_dim];
        for j in (i + 1)..n {
            let b = &k_rows[j * head_dim..(j + 1) * head_dim];
            let c = cosine_similarity(a, b);
            r[i] += c;
            r[j] += c;
            if c > nn_max[i] {
                nn_max[i] = c;
            }
            if c > nn_max[j] {
                nn_max[j] = c;
            }
            pair_sum += c as f64;
            pair_count += 1;
        }
    }

    let inv = 1.0 / (n - 1) as f32; // 대각 제외 평균
    for v in r.iter_mut() {
        *v *= inv;
    }

    let mpc = if pair_count > 0 {
        (pair_sum / pair_count as f64) as f32
    } else {
        0.0
    };
    let redundant = nn_max.iter().filter(|&&c| c > tau).count();
    let redundant_fraction = redundant as f32 / n as f32;

    (
        r,
        RedundancyStats {
            mpc,
            redundant_fraction,
        },
    )
}

/// row-mean R 을 softmax 로 정규화(in-place). 설계서 §2.1 "softmax 정규화 = R" 정의의 정본.
/// numerically-stable(max-shift). 단위 테스트 + 측정 schedule R 분포 덤프에서 사용.
pub(crate) fn softmax_in_place(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// 최근 α 항상 보존 + 나머지 budget 을 Z 상위 single-shot top-k 로 채워 **ascending keep** 산출.
///
/// `z`: per-token fusion 점수(높을수록 보존 가치 높음). `n`: 현 토큰 수. `target`: 보존 토큰 수.
/// `recent_alpha`: 항상 보존하는 최근 토큰 수. target < recent_alpha 면 최근 target 개만 보존.
pub(crate) fn select_keep(z: &[f32], n: usize, target: usize, recent_alpha: usize) -> Vec<usize> {
    let target = target.min(n);
    if target == 0 {
        return Vec::new();
    }
    // 최근 α window: [n - α, n). target 이 더 작으면 최근 target 개만.
    let recent_start = n.saturating_sub(recent_alpha.min(target));
    let mut kept = vec![false; n];
    for slot in kept.iter_mut().skip(recent_start) {
        *slot = true;
    }
    let recent_count = n - recent_start;

    // 나머지 budget 을 최근 window 밖에서 Z 상위로 채운다(single-shot top-k).
    let remaining = target.saturating_sub(recent_count);
    if remaining > 0 {
        let mut candidates: Vec<usize> = (0..recent_start).collect();
        // Z 내림차순(동점은 위치 오름차순으로 결정성 보장).
        candidates.sort_unstable_by(|&a, &b| {
            z[b].partial_cmp(&z[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for &p in candidates.iter().take(remaining) {
            kept[p] = true;
        }
    }

    (0..n).filter(|&p| kept[p]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 동일 벡터 N개 → 모든 pairwise cosine ≈ 1 → R 균등(≈1), MPC≈1, redundant_fraction=1.
    #[test]
    fn row_mean_identical_vectors_uniform_high() {
        let n = 4;
        let head_dim = 3;
        let mut k = Vec::new();
        for _ in 0..n {
            k.extend_from_slice(&[1.0, 2.0, 3.0]);
        }
        let (r, stats) = redundancy_row_mean(&k, n, head_dim, 0.5);
        for (t, &v) in r.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-5, "동일 벡터 R[{t}]≈1, got {v}");
        }
        assert!((stats.mpc - 1.0).abs() < 1e-5, "MPC≈1, got {}", stats.mpc);
        assert!(
            (stats.redundant_fraction - 1.0).abs() < 1e-6,
            "전부 redundant, got {}",
            stats.redundant_fraction
        );
    }

    /// 직교 벡터(서로 cosine 0) → R 균등(≈0), MPC≈0, redundant_fraction=0(τ=0.5).
    #[test]
    fn row_mean_orthogonal_vectors_low() {
        // 3개 표준기저 벡터(서로 직교).
        let k = vec![
            1.0, 0.0, 0.0, // t0
            0.0, 1.0, 0.0, // t1
            0.0, 0.0, 1.0, // t2
        ];
        let (r, stats) = redundancy_row_mean(&k, 3, 3, 0.5);
        for (t, &v) in r.iter().enumerate() {
            assert!(v.abs() < 1e-6, "직교 R[{t}]≈0, got {v}");
        }
        assert!(stats.mpc.abs() < 1e-6, "MPC≈0, got {}", stats.mpc);
        assert!(
            stats.redundant_fraction.abs() < 1e-6,
            "redundant 0, got {}",
            stats.redundant_fraction
        );
    }

    /// 부분 중복: t0≈t1(중복쌍), t2 직교 → t0,t1 의 NN cosine > τ, t2 는 ≤ τ.
    /// redundant_fraction = 2/3.
    #[test]
    fn row_mean_partial_redundancy_fraction() {
        let k = vec![
            1.0, 0.0, 0.0, // t0
            0.99, 0.01, 0.0, // t1 ≈ t0
            0.0, 1.0, 0.0, // t2 직교
        ];
        let (_r, stats) = redundancy_row_mean(&k, 3, 3, 0.5);
        // t0/t1 nearest-neighbour cosine ≈ 0.9999 > 0.5; t2 의 NN(=max(0, ~0.01))< 0.5.
        assert!(
            (stats.redundant_fraction - 2.0 / 3.0).abs() < 1e-5,
            "redundant_fraction=2/3, got {}",
            stats.redundant_fraction
        );
    }

    /// N=1 → R=[0], stats 0(엣지).
    #[test]
    fn row_mean_single_token() {
        let k = vec![1.0, 2.0];
        let (r, stats) = redundancy_row_mean(&k, 1, 2, 0.5);
        assert_eq!(r, vec![0.0]);
        assert_eq!(stats.mpc, 0.0);
        assert_eq!(stats.redundant_fraction, 0.0);
    }

    /// softmax: 균등 입력 → 균등 출력(합=1).
    #[test]
    fn softmax_uniform() {
        let mut v = vec![2.0, 2.0, 2.0, 2.0];
        softmax_in_place(&mut v);
        for &x in &v {
            assert!((x - 0.25).abs() < 1e-6, "균등 softmax=0.25, got {x}");
        }
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    /// softmax: 단조성 — 큰 입력이 큰 출력.
    #[test]
    fn softmax_monotone() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax_in_place(&mut v);
        assert!(v[0] < v[1] && v[1] < v[2], "softmax 단조 보존");
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    /// select_keep: 최근 α 보존 + 나머지 Z top-k. ascending 보장.
    #[test]
    fn select_keep_recent_plus_topk() {
        // n=10, target=5, α=2 → 최근 2(pos 8,9) 보존 + 나머지 3 을 Z top-k(pos 0..8).
        let z = vec![0.0, 9.0, 0.0, 8.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0];
        let keep = select_keep(&z, 10, 5, 2);
        assert_eq!(keep, vec![1, 3, 5, 8, 9], "Z top-3(1,3,5) + 최근2(8,9)");
        assert!(keep.windows(2).all(|w| w[0] < w[1]), "ascending");
    }

    /// select_keep: target < α → 최근 target 개만.
    #[test]
    fn select_keep_target_below_alpha() {
        let z = vec![5.0; 10];
        let keep = select_keep(&z, 10, 3, 8);
        assert_eq!(keep, vec![7, 8, 9], "target=3 < α=8 → 최근 3개");
    }

    /// select_keep: target >= n → 전부 보존(plan 상위에서 None 처리하나 함수 자체 안전성).
    #[test]
    fn select_keep_full() {
        let z = vec![1.0; 4];
        let keep = select_keep(&z, 4, 4, 2);
        assert_eq!(keep, vec![0, 1, 2, 3]);
    }
}
