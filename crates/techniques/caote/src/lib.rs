//! CAOTE technique crate — value-aware KV eviction (attention-output error criticality).
//!
//! 의 동기 사례: 기법이 **자기 metric 을 직접 계산**하는 것을 증명한다(metric 작성자, 선택자
//! 아님). 토큰 criticality = `a_i · ‖v_i − o_h‖` (o_h = Σ_j a_j·v_j, per kv_head). 가중치 `a_i` 는
//! `attn_weight`(있으면) 또는 `importance` 폴백. **value(V)** 를 [`StageCtx::tensor`]`(Value)` 로 읽어
//! plugin 안에서 계산하고, 엔진은 V/weight 노출 + 반환 plan 실행만 한다(plan-returning, D1).
//!
//! `argus-extension-api` 에만 의존(엔진 타입 `KVCache`/`Backend` 미참조). 등록은 `#[distributed_slice]`,
//! 활성화는 force-link 1줄. v1 은 [`KeepSpec::LayerWide`] 만 산출(head reduce 는 plugin 내부;
//! per-head 는 단계 ⑤ executor 대기). feature `caote` 설치 시 `eviction caote` 서브커맨드로 선택.

use argus_extension_api::{
    CacheHandle, CacheOpError, KVMutationStage, MutationPhase, StageCaps, StageCtx, TensorKind,
    register_kv_mutation_stage,
};

/// The score-based caps for the v3 registration: CAOTE weights criticality by the attention weight
/// `a_i` (AttnWeights) and reads cached V (Value), falling back to flat importance (Scores);
/// protects 4 sinks by default.
const CAOTE_CAPS: StageCaps = StageCaps {
    reads: &[
        TensorKind::Scores,
        TensorKind::Value,
        TensorKind::AttnWeights,
    ],
    default_protected_prefix: 4,
    produces_merge_plan: false,
    whole_model: false,
    prefill_attn_window: None,
};

/// CAOTE eviction stage — value-aware criticality.
struct Caote;

/// `a_i` 가중치: per-head attention weight(가용 시) 또는 flat importance 폴백.
fn weight(ctx: &dyn StageCtx, use_aw: bool, kv_head: usize, pos: usize) -> f32 {
    if use_aw {
        ctx.attn_weight(kv_head, pos)
    } else {
        ctx.importance()
            .map_or(0.0, |s| s.get(pos).copied().unwrap_or(0.0))
    }
}

impl Caote {
    /// The value-aware criticality keep-list (`None` = no-op within budget), used by the v3
    /// `on_phase`. CAOTE uses its OWN unstable sort (NOT `keep_top_k` — the tie-break differs):
    /// top-`tgt` criticality, then ascending.
    fn compute_keep(&self, ctx: &dyn StageCtx) -> Option<Vec<usize>> {
        let cur = ctx.current_pos();
        let tgt = ctx.target_len();
        if cur <= tgt {
            return None; // 축소 불필요 — no-op
        }
        let hd = ctx.head_dim();
        let kvh = ctx.n_kv_heads().max(1);
        let has_value = ctx.tensor(TensorKind::Value).is_some();
        let use_aw = ctx.has_attn_weights();

        let mut crit = vec![0.0f32; cur];

        if has_value {
            // value-aware: per kv_head 로 o_h 를 구하고 attention-output 오차로 criticality 누적.
            let mut o = vec![0.0f32; hd];
            let mut v_i = vec![0.0f32; hd];
            for h in 0..kvh {
                // 가중치는 head 마다 1회만 산출(pass1·pass2 공유) — 중복 vtable 호출 제거.
                let w: Vec<f32> = (0..cur).map(|i| weight(ctx, use_aw, h, i)).collect();
                // pass 1: o_h = Σ_i a_i · v_i
                o.iter_mut().for_each(|x| *x = 0.0);
                for (i, &a) in w.iter().enumerate() {
                    if a == 0.0 {
                        continue;
                    }
                    ctx.dequant_v(i, h, &mut v_i);
                    for d in 0..hd {
                        o[d] += a * v_i[d];
                    }
                }
                // pass 2: crit_i += a_i · ‖v_i − o_h‖
                for (i, c) in crit.iter_mut().enumerate() {
                    ctx.dequant_v(i, h, &mut v_i);
                    let mut s = 0.0f32;
                    for d in 0..hd {
                        let e = v_i[d] - o[d];
                        s += e * e;
                    }
                    *c += w[i] * s.sqrt();
                }
            }
        } else {
            // value-unaware 엔진 폴백: weight 합만으로 랭킹(H2O-유사 degrade).
            for (i, c) in crit.iter_mut().enumerate() {
                *c = (0..kvh).map(|h| weight(ctx, use_aw, h, i)).sum();
            }
        }

        // top-`tgt` criticality → ascending keep list (엔진이 new_pos = keep.len() 도출).
        let mut idx: Vec<usize> = (0..cur).collect();
        idx.sort_unstable_by(|&a, &b| {
            crit[b]
                .partial_cmp(&crit[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(tgt);
        idx.sort_unstable();
        Some(idx)
    }
}

// ── v3 native (imperative) surface — the production path ──

impl KVMutationStage for Caote {
    fn name(&self) -> &str {
        "caote"
    }

    /// Stage the value-aware criticality keep-set (or no-op within budget). The driver supplies V via
    /// `ctx.tensor(Value)` (P0-3c V snapshot, armed by `CAOTE_CAPS.reads ∋ Value`) and the attention
    /// weight / importance via `ctx.attn_weight` / `ctx.importance`. Computed via the shared
    /// `compute_keep`.
    fn on_phase(
        &self,
        ctx: &dyn StageCtx,
        cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        match self.compute_keep(ctx) {
            Some(keep) => cache.keep(&keep),
            None => Ok(()),
        }
    }
}

register_kv_mutation_stage!(
    "caote",
    |_p, _args| Box::new(Caote),
    CAOTE_CAPS,
    MutationPhase::KvMutate
);

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{TensorDtype, TensorHandle, TensorShape};

    /// head_dim=2 V handle (one [f32; 2] row per position).
    struct VHandle {
        rows: Vec<[f32; 2]>,
    }
    impl TensorHandle for VHandle {
        fn shape(&self) -> TensorShape {
            TensorShape {
                rows: self.rows.len(),
                cols: 2,
                per_head: true,
            }
        }
        fn dtype(&self) -> TensorDtype {
            TensorDtype::F32
        }
        fn read_row(&self, row: usize, _kv_head: usize, out: &mut [f32]) {
            out.copy_from_slice(&self.rows[row]);
        }
    }

    /// per-(kv_head,pos) attention weight handle (cols=1 → CAOTE's `a_i`).
    struct AHandle {
        a: Vec<f32>,
    }
    impl TensorHandle for AHandle {
        fn shape(&self) -> TensorShape {
            TensorShape {
                rows: self.a.len(),
                cols: 1,
                per_head: true,
            }
        }
        fn dtype(&self) -> TensorDtype {
            TensorDtype::F32
        }
        fn read_row(&self, row: usize, _kv_head: usize, out: &mut [f32]) {
            out[0] = self.a[row];
        }
    }

    /// Value + AttnWeights supplied (single kv_head) — the value-aware production path the
    /// handshake activates. `importance()` is `None` to prove `a_i` comes from AttnWeights.
    struct ValueAwareCtx {
        cur: usize,
        tgt: usize,
        v: VHandle,
        a: AHandle,
    }
    impl StageCtx for ValueAwareCtx {
        fn current_pos(&self) -> usize {
            self.cur
        }
        fn target_len(&self) -> usize {
            self.tgt
        }
        fn layer_idx(&self) -> usize {
            0
        }
        fn importance(&self) -> Option<&[f32]> {
            None
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            2
        }
        fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
            match kind {
                TensorKind::Value => Some(&self.v),
                TensorKind::AttnWeights => Some(&self.a),
                _ => None,
            }
        }
    }

    /// A mock [`CacheHandle`] capturing the keep staged by `keep`.
    struct CaptureHandle {
        kept: Option<Vec<usize>>,
    }
    impl CacheHandle for CaptureHandle {
        fn current_pos(&self) -> usize {
            4
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            2
        }
        fn kv_on_device(&self) -> bool {
            false
        }
        fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
            None
        }
        fn keep(&mut self, keep: &[usize]) -> Result<(), CacheOpError> {
            self.kept = Some(keep.to_vec());
            Ok(())
        }
        fn keep_per_head(&mut self, _keep: &[&[usize]]) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn merge(
            &mut self,
            _merges: &[argus_extension_api::WeightedMerge],
        ) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn reencode(&mut self, _target: argus_extension_api::FormatId) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn transition_quant_bits(&mut self, _bits: u8) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn offload(&mut self, _prefix_len: usize) -> Result<(), CacheOpError> {
            Ok(())
        }
        fn recall(&mut self) -> Result<(), CacheOpError> {
            Ok(())
        }
    }

    /// v3 native registration + DECISION: the v3 `on_phase` stages exactly the value-aware
    /// criticality keep-set via the shared `compute_keep` (which uses CAOTE's own unstable sort,
    /// NOT keep_top_k).
    #[test]
    fn v3_native_value_aware_decision() {
        use argus_extension_api::find_mutation_stage;
        let reg = find_mutation_stage("caote").expect("caote in KV_MUTATION_STAGES");
        assert_eq!(reg.name, "caote");
        assert_eq!(reg.caps, CAOTE_CAPS);
        assert_eq!(
            (reg.make)(argus_extension_api::StageParams::default(), &[]).name(),
            "caote"
        );

        // a_i = [0.1,0.2,0.3,0.4]; v = [[1,0],[0,1],[1,1],[2,2]]; o_h = Σ a_i·v_i = [1.2,1.3];
        // crit_i = a_i·‖v_i − o_h‖ → top-2 = {3,1} → keep [1,3]. Weight-only fallback would keep
        // [2,3], so asserting [1,3] proves the value-aware (Value + AttnWeights) path ran.
        let ctx = ValueAwareCtx {
            cur: 4,
            tgt: 2,
            v: VHandle {
                rows: vec![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
            },
            a: AHandle {
                a: vec![0.1, 0.2, 0.3, 0.4],
            },
        };
        let mut h = CaptureHandle { kept: None };
        <Caote as KVMutationStage>::on_phase(&Caote, &ctx, &mut h).unwrap();
        assert_eq!(h.kept, Some(vec![1, 3]), "value-aware keep");
    }
}
