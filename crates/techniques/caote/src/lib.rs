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
    KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, KeepSpec, StageCaps, StageCtx,
    StageParams, TensorKind,
};
use linkme::distributed_slice;

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

impl KVCacheStage for Caote {
    fn name(&self) -> &str {
        "caote"
    }

    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
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
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(idx),
            merges: Vec::new(),
        })
    }
}

/// 등록 — 엔진은 construction 시 `find_stage("caote")` 로 이 항목을 찾는다.
#[distributed_slice(KV_CACHE_STAGES)]
static CAOTE: KVCacheStageReg = KVCacheStageReg {
    name: "caote",
    make: |_params: StageParams| Box::new(Caote),
    // caote takes no technique-private args — drop the blob.
    make_with_args: |_params: StageParams, _args| Box::new(Caote),
    // CAOTE weights criticality by the attention weight a_i when available
    // (ctx.attn_weight / has_attn_weights => AttnWeights) and reads cached V
    // (ctx.dequant_v => Value), falling back to flat importance (Scores). Declare
    // all three so the buffer-allocation handshake can wire each; it is score-based
    // (non-empty reads). Protect 4 attention sinks by default.
    caps: StageCaps {
        reads: &[
            argus_extension_api::TensorKind::Scores,
            argus_extension_api::TensorKind::Value,
            argus_extension_api::TensorKind::AttnWeights,
        ],
        default_protected_prefix: 4,
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{TensorDtype, TensorHandle, TensorShape, find_stage};

    /// V 미공급 mock — fallback(importance 랭킹) 경로 검증용.
    struct MockCtx {
        cur: usize,
        tgt: usize,
        imp: Vec<f32>,
    }
    impl StageCtx for MockCtx {
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
            Some(&self.imp)
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            4
        }
        fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
            None // value-unaware → CAOTE fallback
        }
    }

    fn params() -> StageParams {
        StageParams {
            eviction_window: 0,
            protected_prefix: 0,
            keep_ratio: 0.0,
            sink_size: 0,
            streaming_window: 0,
        }
    }

    #[test]
    fn registers_into_slice() {
        let reg = find_stage("caote").expect("caote 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "caote");
    }

    #[test]
    fn fallback_ranks_by_importance_without_value() {
        let stage = (find_stage("caote").unwrap().make)(params());
        assert_eq!(stage.name(), "caote");
        // importance: pos 1,3 이 최대 → tgt=2 면 {1,3} 유지(ascending).
        let ctx = MockCtx {
            cur: 5,
            tgt: 2,
            imp: vec![0.1, 9.0, 0.2, 8.0, 0.3],
        };
        let plan = stage.plan(&ctx).expect("plan Some");
        match plan.keep {
            KeepSpec::LayerWide(k) => assert_eq!(k, vec![1, 3]),
            KeepSpec::PerHead(_) => panic!("v1 은 LayerWide"),
        }
        assert!(plan.merges.is_empty());
        // current <= target → no-op.
        assert!(
            stage
                .plan(&MockCtx {
                    cur: 2,
                    tgt: 5,
                    imp: vec![],
                })
                .is_none()
        );
    }

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

    #[test]
    fn value_aware_criticality_uses_attn_weights_and_value() {
        // a_i = [0.1, 0.2, 0.3, 0.4]; v = [[1,0],[0,1],[1,1],[2,2]].
        // o_h = Σ a_i·v_i = [1.2, 1.3]; crit_i = a_i·‖v_i − o_h‖ →
        //   [0.1315, 0.2474, 0.1082, 0.4252]. top-2 = {3,1} → keep [1,3].
        // Weight-only fallback would rank by a_i → {3,2} → keep [2,3]; asserting [1,3] proves
        // the value-aware (Value + AttnWeights) path actually ran.
        let stage = (find_stage("caote").unwrap().make)(params());
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
        let plan = stage.plan(&ctx).expect("plan Some");
        match plan.keep {
            KeepSpec::LayerWide(k) => assert_eq!(k, vec![1, 3], "value-aware keep-set"),
            KeepSpec::PerHead(_) => panic!("v1 은 LayerWide"),
        }
        assert!(plan.merges.is_empty());
    }
}
