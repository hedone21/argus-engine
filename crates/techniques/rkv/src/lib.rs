//! R-KV measurement technique crate — cosine redundancy + importance joint eviction stage.
//!
//! arXiv 2505.24133 (NeurIPS'25). KV roadmap 항목 0 측정 스프린트(P2a):
//! `arch/kv_roadmap_item0_measurement.md` §2.1 / §4.1.
//!
//! Extracted from the engine core into a self-registering technique crate (the
//! `streaming-llm`/`h2o`/`d2o` precedent): depends only on `argus-extension-api` + `linkme`,
//! implements [`KVCacheStage`], and registers under the name `"rkv"` via
//! `#[distributed_slice(KV_CACHE_STAGES)]`. The engine force-links it under the `rkv` feature with
//! `#[cfg(feature = "rkv")] use rkv as _;` and finds it via `find_stage("rkv")` — feature OFF =
//! unlinked + `eviction rkv` subcommand absent (production catalog unchanged).
//!
//! **수식** (per-KV-head, §2.1):
//! 1. redundancy R: K(key) pairwise cosine N×N 행렬의 row-mean → softmax 정규화.
//! 2. importance I: 기존 accumulator score 재사용(설계서 근사 허용), 최근 α window.
//! 3. fusion Z = λ·I − (1−λ)·R, **λ=0.1**(redundancy 지배).
//! 4. 최근 α=8 항상 보존 + 나머지 budget 을 Z 상위 single-shot top-k.
//!
//! **GQA**: redundancy/importance 모두 KV-head 단위로 측정. head 별 Z 를 평균해 단일 layer-wide
//! keep 산출(per-head 차등 keep 은 별도 영역 — 본 프로토타입은 layer-wide 근사).
//!
//! **재사용**(§4.1): K 읽기는 `StageCtx::dequant_k`(엔진 `dequantize_k` 정본 위임), cosine 은 이
//! crate 가 자체 보유한 [`cosine_similarity`](d2o 선례). N×N row-mean 집계 루프만 신규.
//!
//! λ 는 CLI(`eviction rkv --lambda`)에서 opaque [`StageArgs`] blob(`lambda` 키)으로 흘러들어와
//! [`RkvStage`]가 직접 파싱한다 — 엔진은 rkv 의 private 파라미터를 모른다.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use argus_extension_api::{
    KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, KeepSpec, StageArgs, StageCaps,
    StageCtx, StageParams, TensorKind,
};
use linkme::distributed_slice;

/// 1단 측정 덤프 게이트 env var. set 시 plan() 마다 per-kv_head `[RkvStats]` 마커 라인을 stderr 로
/// 출력한다(파싱 가능 포맷). 측정 전용 — 미설정 시 덤프 경로 미진입(production 무영향).
const RKV_DUMP_ENV: &str = "ARGUS_RKV_DUMP";

/// fusion 가중치 λ (Z = λ·I − (1−λ)·R). 기본 0.1 = redundancy 지배(논문 §2.1).
pub const RKV_DEFAULT_LAMBDA: f32 = 0.1;
/// redundant fraction 의 nearest-neighbour cosine 임계 τ (§5 표, D2O cosine 정합).
pub const RKV_TAU: f32 = 0.5;
/// 항상 보존하는 최근 토큰 수 α (importance window 겸용).
pub const RKV_RECENT_ALPHA: usize = 8;

/// 두 벡터의 코사인 유사도. 엔진 `kv::dequant::cosine_similarity` 와 bit-identical(plugin 은 엔진을
/// 참조할 수 없으므로 d2o 선례대로 자체 보유한다).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

/// R-KV 측정 전용 설정(CLI 노출은 λ 만 — α/τ 는 측정 상수).
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

impl RkvConfig {
    /// opaque [`StageArgs`] blob 에서 λ(`lambda` 키)를 파싱한다. 나머지 키는 무시, α/τ 는 측정 상수.
    /// 엔진은 이 키를 모르고 `eviction rkv --lambda <L>` 의 값을 blob 으로만 전달한다.
    pub fn from_args(args: StageArgs<'_>) -> Self {
        let mut cfg = Self::default();
        for a in args.iter().filter(|a| a.key == "lambda") {
            if let Ok(v) = a.val.parse::<f32>() {
                cfg.lambda = v;
            }
        }
        cfg
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
/// 측정 hook 이 마지막 계산 stats 를 보관하도록 `Mutex<Vec<RedundancyStats>>`를 둔다 — 측정 schedule
/// 이 plan() 호출 후 [`last_stats`](Self::last_stats)로 읽는다(stderr/CSV 덤프).
pub struct RkvStage {
    config: RkvConfig,
    /// 마지막 plan() 의 per-kv_head redundancy stats (측정 1단 덤프용). plan 은 `&self` 라 내부가변.
    last_stats: Mutex<Vec<RedundancyStats>>,
    /// plan() 호출 순번 — `[RkvStats]` 덤프의 `layer=` 필드(누적 카운터, 측정 전용).
    plan_calls: AtomicUsize,
}

impl RkvStage {
    /// 주어진 설정으로 생성.
    pub fn new(config: RkvConfig) -> Self {
        Self {
            config,
            last_stats: Mutex::new(Vec::new()),
            plan_calls: AtomicUsize::new(0),
        }
    }

    /// 직전 plan() 이 계산한 per-kv_head redundancy stats 의 복사본(측정 1단 덤프 진입점).
    pub fn last_stats(&self) -> Vec<RedundancyStats> {
        self.last_stats.lock().expect("rkv stats poisoned").clone()
    }
}

/// per-kv_head redundancy stats 를 `[RkvStats]` 마커 라인으로 stderr 덤프(env `ARGUS_RKV_DUMP` 게이트).
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
        // K 텐서가 없으면(score-free 엔진) redundancy 계산 불가 → no-op.
        ctx.tensor(TensorKind::Key)?;

        let n_kv_heads = ctx.n_kv_heads().max(1);
        let head_dim = ctx.head_dim();
        let importance = ctx.importance();

        let mut k_rows = vec![0.0f32; n * head_dim]; // 재사용 버퍼(per head)
        let mut z_sum = vec![0.0f32; n]; // head 합산 Z
        let mut stats = Vec::with_capacity(n_kv_heads);

        for h in 0..n_kv_heads {
            for (t, row) in k_rows.chunks_mut(head_dim).enumerate().take(n) {
                ctx.dequant_k(t, h, row);
            }
            let (mut r, stat) = redundancy_row_mean(&k_rows, n, head_dim, self.config.tau);
            softmax_in_place(&mut r);
            for t in 0..n {
                let i = importance
                    .and_then(|imp| imp.get(t).copied())
                    .unwrap_or(0.0);
                z_sum[t] += self.config.lambda * i - (1.0 - self.config.lambda) * r[t];
            }
            stats.push(stat);
        }

        let inv_heads = 1.0 / n_kv_heads as f32;
        for z in z_sum.iter_mut() {
            *z *= inv_heads;
        }

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

/// row-mean R 을 softmax 로 정규화(in-place). numerically-stable(max-shift).
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
pub(crate) fn select_keep(z: &[f32], n: usize, target: usize, recent_alpha: usize) -> Vec<usize> {
    let target = target.min(n);
    if target == 0 {
        return Vec::new();
    }
    let recent_start = n.saturating_sub(recent_alpha.min(target));
    let mut kept = vec![false; n];
    for slot in kept.iter_mut().skip(recent_start) {
        *slot = true;
    }
    let recent_count = n - recent_start;

    let remaining = target.saturating_sub(recent_count);
    if remaining > 0 {
        let mut candidates: Vec<usize> = (0..recent_start).collect();
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

/// Registration — `find_stage("rkv")`. λ rides the opaque [`StageArgs`] blob (`lambda` key);
/// α/τ are measurement constants. Score-based (fusion includes the importance term).
#[distributed_slice(KV_CACHE_STAGES)]
static RKV: KVCacheStageReg = KVCacheStageReg {
    name: "rkv",
    make: |_p: StageParams| Box::new(RkvStage::new(RkvConfig::default())),
    make_with_args: |_p: StageParams, args| Box::new(RkvStage::new(RkvConfig::from_args(args))),
    // R-KV's fusion Z = λ·I − (1−λ)·R includes the importance term, so it is score-based; protect 4
    // attention sinks by default (same as the other score-based stages).
    caps: StageCaps {
        reads: &[argus_extension_api::TensorKind::Scores],
        default_protected_prefix: 4,
        produces_merge_plan: false,
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{PluginArg, TensorDtype, TensorHandle, TensorShape, find_stage};

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

    /// 부분 중복: t0≈t1, t2 직교 → redundant_fraction = 2/3.
    #[test]
    fn row_mean_partial_redundancy_fraction() {
        let k = vec![
            1.0, 0.0, 0.0, // t0
            0.99, 0.01, 0.0, // t1 ≈ t0
            0.0, 1.0, 0.0, // t2 직교
        ];
        let (_r, stats) = redundancy_row_mean(&k, 3, 3, 0.5);
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

    #[test]
    fn softmax_uniform() {
        let mut v = vec![2.0, 2.0, 2.0, 2.0];
        softmax_in_place(&mut v);
        for &x in &v {
            assert!((x - 0.25).abs() < 1e-6, "균등 softmax=0.25, got {x}");
        }
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_monotone() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax_in_place(&mut v);
        assert!(v[0] < v[1] && v[1] < v[2], "softmax 단조 보존");
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn select_keep_recent_plus_topk() {
        let z = vec![0.0, 9.0, 0.0, 8.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0];
        let keep = select_keep(&z, 10, 5, 2);
        assert_eq!(keep, vec![1, 3, 5, 8, 9], "Z top-3(1,3,5) + 최근2(8,9)");
        assert!(keep.windows(2).all(|w| w[0] < w[1]), "ascending");
    }

    #[test]
    fn select_keep_target_below_alpha() {
        let z = vec![5.0; 10];
        let keep = select_keep(&z, 10, 3, 8);
        assert_eq!(keep, vec![7, 8, 9], "target=3 < α=8 → 최근 3개");
    }

    #[test]
    fn select_keep_full() {
        let z = vec![1.0; 4];
        let keep = select_keep(&z, 4, 4, 2);
        assert_eq!(keep, vec![0, 1, 2, 3]);
    }

    #[test]
    fn registers_with_score_based_caps() {
        let reg = find_stage("rkv").expect("rkv registered in KV_CACHE_STAGES");
        assert_eq!(reg.name, "rkv");
        assert!(!reg.caps.reads.is_empty(), "rkv fusion includes importance");
        assert_eq!(reg.caps.default_protected_prefix, 4);
    }

    #[test]
    fn from_args_parses_lambda() {
        let cfg = RkvConfig::from_args(&[PluginArg {
            key: "lambda",
            val: "0.42",
        }]);
        assert!((cfg.lambda - 0.42).abs() < 1e-6);
        // unknown keys ignored; α/τ unchanged.
        let cfg2 = RkvConfig::from_args(&[PluginArg {
            key: "nope",
            val: "x",
        }]);
        assert!((cfg2.lambda - RKV_DEFAULT_LAMBDA).abs() < 1e-6);
    }

    /// 단일-head K 행렬을 공급하는 최소 ctx — `dequant_k`(Key 핸들) + importance 로 plan 산출 검증.
    struct KCtx {
        n: usize,
        head_dim: usize,
        k: Vec<f32>, // [n * head_dim]
        target: usize,
        importance: Vec<f32>,
    }
    struct KHandle<'a> {
        k: &'a [f32],
        n: usize,
        head_dim: usize,
    }
    impl TensorHandle for KHandle<'_> {
        fn shape(&self) -> TensorShape {
            TensorShape {
                rows: self.n,
                cols: self.head_dim,
                per_head: true,
            }
        }
        fn dtype(&self) -> TensorDtype {
            TensorDtype::F32
        }
        fn read_row(&self, row: usize, _kv_head: usize, out: &mut [f32]) {
            out.copy_from_slice(&self.k[row * self.head_dim..(row + 1) * self.head_dim]);
        }
    }
    impl StageCtx for KCtx {
        fn current_pos(&self) -> usize {
            self.n
        }
        fn target_len(&self) -> usize {
            self.target
        }
        fn layer_idx(&self) -> usize {
            0
        }
        fn importance(&self) -> Option<&[f32]> {
            Some(&self.importance)
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            self.head_dim
        }
        fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
            match kind {
                TensorKind::Key => Some(Box::leak(Box::new(KHandle {
                    k: &self.k,
                    n: self.n,
                    head_dim: self.head_dim,
                })) as &dyn TensorHandle),
                _ => None,
            }
        }
    }

    #[test]
    fn plan_keeps_target_len_and_dumps_stats() {
        // 8 tokens, distinct K per token, target=4 → keep 4 ascending, last_stats populated.
        let head_dim = 3;
        let n = 8;
        let mut k = vec![0.0f32; n * head_dim];
        for t in 0..n {
            k[t * head_dim] = t as f32; // distinct per token
        }
        let ctx = KCtx {
            n,
            head_dim,
            k,
            target: 4,
            importance: vec![1.0; n],
        };
        let stage = RkvStage::new(RkvConfig::default());
        let plan = stage.plan(&ctx).expect("rkv plan Some (target<n)");
        match plan.keep {
            KeepSpec::LayerWide(keep) => {
                assert_eq!(keep.len(), 4, "target_len=4 만큼 보존");
                assert!(keep.windows(2).all(|w| w[0] < w[1]), "ascending");
            }
            KeepSpec::PerHead(_) => panic!("rkv prototype is layer-wide"),
        }
        let stats = stage.last_stats();
        assert_eq!(stats.len(), 1, "kv_heads=1 → stats 1개");
        assert!(stats[0].mpc.is_finite());
        assert!((0.0..=1.0).contains(&stats[0].redundant_fraction));
    }
}
