//! 번들 예제 plugin — 한 `.so` 가 **stage 1 + format 1** 을 동시에 export. 작성자는
//! `register_kv_stage!` + `register_kv_format!` 를 한 crate 에서 호출하고 `export_plugin!()` 1회로
//! 양축 엔트리(register_kv_stages_v2 ⊥ register_kv_formats_v2)를 emit 한다.
//!
//! host dispatcher(`register_dynamic_plugins`)는 이 `.so` 를 1회 dlopen 해 stage-reg·format-reg 를
//! 동일 `Arc<Library>` 공유로 양축 registry 에 분리 등록한다(병합 없음). "축별 `.so` 분리"
//! 가 불필요함을 실증하는 vehicle(번들 양축 등록).

use argus_extension_api::{
    CacheHandle, CacheOpError, KVCachePlan, KVCacheStage, KVFormat, KVLayoutDesc, KVMutationStage,
    KeepSpec, MutationPhase, Packing, ScaleLayout, StageCtx, StageParams,
};

/// 번들 stage — 최근 `target_len` 토큰 유지(example_keep_recent 와 동형, 다른 이름).
struct BundleKeep;
impl BundleKeep {
    fn keep_list(&self, current: usize, target: usize) -> Option<Vec<usize>> {
        (current > target).then(|| (current - target..current).collect())
    }
}
// v3 native (the production path) + v2 plan-returning (migration window). Both via `keep_list`.
impl KVMutationStage for BundleKeep {
    fn name(&self) -> &str {
        "bundle_keep"
    }
    fn on_phase(
        &self,
        ctx: &dyn StageCtx,
        cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        match self.keep_list(ctx.current_pos(), ctx.target_len()) {
            Some(keep) => cache.keep(&keep),
            None => Ok(()),
        }
    }
}
impl KVCacheStage for BundleKeep {
    fn name(&self) -> &str {
        "bundle_keep"
    }
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        self.keep_list(ctx.current_pos(), ctx.target_len())
            .map(|keep| KVCachePlan {
                keep: KeepSpec::LayerWide(keep),
                merges: Vec::new(),
                channels: None,
            })
    }
}

/// per-head keep 을 산출하는 stage — 한 `.so` 에 stage 2종(bundle_keep LayerWide + bundle_perhead
/// PerHead) = 멀티-stage 인덱스 바인딩 검증. v3 routes the per-head keep through `keep_per_head`.
struct BundlePerHead;
impl BundlePerHead {
    fn per_head_keep(
        &self,
        current: usize,
        target: usize,
        n_kv_heads: usize,
    ) -> Option<Vec<Vec<usize>>> {
        if current <= target {
            return None;
        }
        let keep: Vec<usize> = (current - target..current).collect();
        Some(vec![keep; n_kv_heads.max(1)]) // all heads keep the same (equal-length invariant)
    }
}
impl KVMutationStage for BundlePerHead {
    fn name(&self) -> &str {
        "bundle_perhead"
    }
    fn on_phase(
        &self,
        ctx: &dyn StageCtx,
        cache: &mut dyn CacheHandle,
    ) -> Result<(), CacheOpError> {
        match self.per_head_keep(ctx.current_pos(), ctx.target_len(), ctx.n_kv_heads()) {
            None => Ok(()),
            Some(heads) => {
                let refs: Vec<&[usize]> = heads.iter().map(|h| h.as_slice()).collect();
                cache.keep_per_head(&refs)
            }
        }
    }
}
impl KVCacheStage for BundlePerHead {
    fn name(&self) -> &str {
        "bundle_perhead"
    }
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        self.per_head_keep(ctx.current_pos(), ctx.target_len(), ctx.n_kv_heads())
            .map(|per_head| KVCachePlan {
                keep: KeepSpec::PerHead(per_head),
                merges: Vec::new(),
                channels: None,
            })
    }
}

/// 번들 format — q4_0-like descriptor.
struct BundleFmt;
impl KVFormat for BundleFmt {
    fn name(&self) -> &str {
        "bundle_fmt"
    }
    fn layout(&self) -> KVLayoutDesc {
        KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        }
    }
}

// 한 crate(=한 `.so`)에 stage 2종 + format 1종 — const-block 격리 다회 호출 + export 1회.
argus_extension_api::register_kv_stage!("bundle_keep", |_p: StageParams| Box::new(BundleKeep));
argus_extension_api::register_kv_stage!("bundle_perhead", |_p: StageParams| Box::new(
    BundlePerHead
));
argus_extension_api::register_kv_format!("bundle_fmt", || Box::new(BundleFmt));
argus_extension_api::export_plugin!();

// v3 native registrations (static-linkme only) for both KV stages — the format half is unchanged.
argus_extension_api::register_kv_mutation_stage!(
    "bundle_keep",
    |_p| Box::new(BundleKeep),
    MutationPhase::KvMutate
);
argus_extension_api::register_kv_mutation_stage!(
    "bundle_perhead",
    |_p| Box::new(BundlePerHead),
    MutationPhase::KvMutate
);

#[cfg(test)]
mod tests {
    use super::*;
    use argus_extension_api::{TensorHandle, TensorKind, find_kv_format, find_stage};

    /// Minimal ctx — both bundle stages read only current_pos / target_len / n_kv_heads.
    struct Ctx {
        cur: usize,
        tgt: usize,
        n_kv: usize,
    }
    impl StageCtx for Ctx {
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
            self.n_kv
        }
        fn head_dim(&self) -> usize {
            1
        }
        fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
            None
        }
    }

    /// A mock [`CacheHandle`] capturing keep / keep_per_head.
    #[derive(Default)]
    struct CaptureHandle {
        kept: Option<Vec<usize>>,
        kept_per_head: Option<Vec<Vec<usize>>>,
    }
    impl CacheHandle for CaptureHandle {
        fn current_pos(&self) -> usize {
            100
        }
        fn n_kv_heads(&self) -> usize {
            2
        }
        fn head_dim(&self) -> usize {
            1
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
        fn keep_per_head(&mut self, keep: &[&[usize]]) -> Result<(), CacheOpError> {
            self.kept_per_head = Some(keep.iter().map(|h| h.to_vec()).collect());
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

    /// DECISION equivalence: each bundle stage's v3 on_phase stages exactly what its v2 plan returns
    /// (BundleKeep -> LayerWide keep; BundlePerHead -> per-head keep), including the no-op case.
    #[test]
    fn bundle_v3_matches_v2_decision() {
        // BundleKeep: LayerWide.
        let ctx = Ctx {
            cur: 100,
            tgt: 30,
            n_kv: 2,
        };
        let v2_keep = match BundleKeep.plan(&ctx).unwrap().keep {
            KeepSpec::LayerWide(k) => k,
            KeepSpec::PerHead(_) => panic!("bundle_keep is layer-wide"),
        };
        let mut h = CaptureHandle::default();
        <BundleKeep as KVMutationStage>::on_phase(&BundleKeep, &ctx, &mut h).unwrap();
        assert_eq!(h.kept, Some(v2_keep));
        // BundlePerHead: PerHead.
        let v2_heads = match BundlePerHead.plan(&ctx).unwrap().keep {
            KeepSpec::PerHead(h) => h,
            KeepSpec::LayerWide(_) => panic!("bundle_perhead is per-head"),
        };
        let mut h2 = CaptureHandle::default();
        <BundlePerHead as KVMutationStage>::on_phase(&BundlePerHead, &ctx, &mut h2).unwrap();
        assert_eq!(h2.kept_per_head, Some(v2_heads));
        // no-op (within budget) stages nothing for either.
        let noop = Ctx {
            cur: 20,
            tgt: 30,
            n_kv: 2,
        };
        let mut h3 = CaptureHandle::default();
        <BundleKeep as KVMutationStage>::on_phase(&BundleKeep, &noop, &mut h3).unwrap();
        assert_eq!(h3.kept, None);
        assert!(BundleKeep.plan(&noop).is_none());
    }

    #[test]
    fn bundle_registers_both_axes() {
        assert_eq!(
            find_stage("bundle_keep").expect("stage 등록").name,
            "bundle_keep"
        );
        assert_eq!(
            find_stage("bundle_perhead")
                .expect("perhead stage 등록")
                .name,
            "bundle_perhead"
        );
        assert_eq!(
            find_kv_format("bundle_fmt").expect("format 등록").name,
            "bundle_fmt"
        );
    }

    /// v3 native: both KV stages register in KV_MUTATION_STAGES (the format half is unchanged). The
    /// keep decision is shared with the v2 plan via `keep_list` / `per_head_keep`, so they are
    /// byte-identical by construction.
    #[test]
    fn bundle_stages_register_in_mutation_slice() {
        use argus_extension_api::{MutationPhase, find_mutation_stage};
        for name in ["bundle_keep", "bundle_perhead"] {
            let reg =
                find_mutation_stage(name).unwrap_or_else(|| panic!("{name} in mutation slice"));
            assert_eq!(reg.name, name);
            assert_eq!(reg.phase, MutationPhase::KvMutate);
            assert_eq!(
                (reg.make)(argus_extension_api::StageParams::default(), &[]).name(),
                name
            );
        }
        // the format axis is untouched by the v3 migration.
        assert!(find_kv_format("bundle_fmt").is_some());
    }
}
