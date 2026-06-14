//! KIVI-quantized KV cache용 Forward 구현체 (Phase 4-5-a).
//!
//! Phase 4-3 [`ModelForward`] 패턴을 복제하되 plan path를 제외한다.
//! KIVI 고유의 `Vec<KiviCache>` + `forward_into` 경로를 `Forward` trait으로 래핑.
//!
//! chat REPL 통합은 Phase 4-5-d `ChatSession`에서 수행. 본 파일은 컴파일
//! 가능한 Forward 구현체를 제공하는 것이 목표.

use std::sync::Arc;

use anyhow::Result;

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::capability::kivi_attention::KiviAttentionBackend;
use crate::format::KVCacheFormat;
use crate::inference::sampling::StepCtx;
use crate::kv::kivi_cache::KiviCache;
use crate::kv::kivi_format::KIVIFormat;
use crate::layers::workspace::{LayerWorkspace, WorkspaceConfig};
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use crate::session::forward::Forward;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// KIVI 양자화 KV cache를 사용하는 Forward 구현체.
///
/// AB-2 §5.7.1 (A안): persistent `Vec<Arc<KIVIFormat>>` 를 보유하며(ModelForward `fmt_caches`
/// 동형 — `new()` 1회 wrap), `TransformerModel::forward_into`를 직접 호출한다. KiviQuantStage 가
/// register 시점 이 handle 을 clone 해 `transition_bits` 를 적용한다(INV-STAGE-LAYER-HANDLE).
/// plan path는 KIVI 경로에서 미사용이므로 포함하지 않는다.
pub struct KiviForward {
    backend: Arc<dyn Backend>,
    cpu_backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    model: Arc<TransformerModel>,
    /// AB-2 §5.7.1 (A안): persistent KIVI handle (전 prefill/step 동안 보존). enumerate 순서 ==
    /// layer idx. `new()` 에서 1회 wrap 하고, prefill/step 은 보유 Arc clone 으로 dyn fmts 만 구성.
    kivi_caches: Vec<Arc<KIVIFormat>>,

    decode_workspace: LayerWorkspace,
    decode_input: Tensor,        // [1, 1] U8 (u32 token id, GPU-side)
    decode_x_gen: Tensor,        // [1, 1, hidden_size] F32
    logits_decode: Tensor,       // [1, 1, vocab_size] F32
    logits_prefill_last: Tensor, // [1, 1, vocab_size] F32

    vocab_size: usize,
    // Phase 4-5-d ChatSession이 stats_line 등에서 사용 예정.
    #[allow(dead_code)]
    bits: u8,
    #[allow(dead_code)]
    residual_size: usize,
}

impl KiviForward {
    /// `KiviForward`를 생성한다.
    ///
    /// `kv_caches`는 [`alloc_kivi_kv_caches`]로 미리 할당해서 전달한다.
    /// `max_seq_len`은 KV cache 할당 시 사용한 것과 동일한 값을 전달한다.
    pub fn new(
        backend: Arc<dyn Backend>,
        memory: Arc<dyn Memory>,
        model: Arc<TransformerModel>,
        kv_caches: Vec<KiviCache>,
        bits: u8,
        residual_size: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let hidden_size = model.config.hidden_size;
        let vocab_size = model.config.vocab_size;

        let decode_workspace = LayerWorkspace::new(
            workspace_config_for(&model, max_seq_len),
            memory.as_ref(),
            backend.clone(),
        )?;

        let decode_input_buf = memory.alloc(4, DType::U8)?;
        let decode_input = Tensor::new(Shape::new(vec![1, 1]), decode_input_buf, backend.clone());

        let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
        let decode_x_gen = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            x_gen_buf,
            backend.clone(),
        );

        let logits_decode = alloc_logits(memory.as_ref(), backend.clone(), vocab_size)?;
        let logits_prefill_last = alloc_logits(memory.as_ref(), backend.clone(), vocab_size)?;

        // AB-2 §5.7.1 (A안): KIVI cache 를 persistent Arc handle 로 1회 wrap (ModelForward
        // `ensure_fmt_wrapped` 동형). enumerate 순서 == layer idx. 이후 prefill/step 은 보유 Arc
        // clone 으로 dyn fmts 만 구성한다(per-step mem::take + try_unwrap 댄스 제거).
        let kivi_caches: Vec<Arc<KIVIFormat>> = kv_caches
            .into_iter()
            .enumerate()
            .map(|(i, c)| Arc::new(KIVIFormat::new(i, c)))
            .collect();

        Ok(Self {
            backend,
            cpu_backend,
            memory,
            model,
            kivi_caches,
            decode_workspace,
            decode_input,
            decode_x_gen,
            logits_decode,
            logits_prefill_last,
            vocab_size,
            bits,
            residual_size,
        })
    }

    /// AB-2 §5.7.1/§5.7.8: persistent KIVI handle 슬라이스 (dispatcher/KiviQuantStage 가
    /// register 시점 `.to_vec()` 으로 clone 보유 — `ModelForward::fmt_caches()` 동형).
    pub fn kivi_caches(&self) -> &[Arc<KIVIFormat>] {
        &self.kivi_caches
    }

    /// prefill용 입력 텐서를 백엔드에 업로드한다.
    fn build_input_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        let seq_len = tokens.len();
        let cpu_buf = Galloc::new().alloc(seq_len * 4, DType::U8)?;
        // SAFETY: Galloc은 64B 정렬 블록을 반환하므로 u32 정렬을 만족한다.
        // seq_len * 4 바이트를 즉시 초기화한다.
        unsafe {
            let dst = cpu_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), dst, seq_len);
        }
        let cpu_tensor = Tensor::new(
            Shape::new(vec![1, seq_len]),
            cpu_buf,
            self.cpu_backend.clone(),
        );
        self.backend.copy_from(&cpu_tensor)
    }

    /// logits 텐서를 백엔드에서 읽어 Vec<f32>로 반환한다.
    fn read_logits(&self, logits: &Tensor) -> Result<Vec<f32>> {
        self.backend.synchronize()?;
        let mut out = vec![0.0f32; self.vocab_size];
        // SAFETY: out은 vocab_size 크기의 초기화된 f32 슬라이스다.
        // read_buffer는 GPU 버퍼의 f32 바이트를 호스트 메모리에 쓴다.
        unsafe {
            let bytes =
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, self.vocab_size * 4);
            self.backend.read_buffer(logits, bytes)?;
        }
        Ok(out)
    }
}

impl Forward for KiviForward {
    fn prefill(&mut self, tokens: &[u32], start_pos: usize) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            anyhow::bail!("KiviForward::prefill received zero tokens");
        }
        let input_tensor = self.build_input_tensor(tokens)?;

        // AB-2 §5.7.1 (A안): persistent KIVI handle 의 Arc clone 으로 dyn fmts 구성 (transient
        // mem::take + try_unwrap 댄스 제거 — handle 영속이라 panic 위험 소멸). AWQE need_scores 는
        // layer-0 handle 에 with_cache_mut 으로 query (concrete `is_awqe_enabled` 위임).
        let need_scores = self
            .kivi_caches
            .first()
            .is_some_and(|h| h.with_cache_mut(|c| c.is_awqe_enabled()));
        let dyn_fmts: Vec<Arc<dyn KVCacheFormat>> = self
            .kivi_caches
            .iter()
            .map(|a| a.clone() as Arc<dyn KVCacheFormat>)
            .collect();

        // borrow 충돌 회피: Arc clone으로 backend/memory 분리.
        let backend = self.backend.clone();
        let memory_ref: *const dyn Memory = self.memory.as_ref();
        // SAFETY: self.memory는 self가 살아있는 동안 유효하다.
        // raw pointer는 현재 stack frame 내에서만 역참조된다.
        let memory: &dyn Memory = unsafe { &*memory_ref };

        let fwd_result = self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos,
            fmts: &dyn_fmts,
            backend: &backend,
            memory,
            logits_out: &mut self.logits_prefill_last,
            x_gen: None,
            workspace: None,
            logits_last_only: true,
            score_accumulator: None,
            query_stats_accumulator: None,
            skip_config: None,
            importance_collector: None,
            cache_self_need_scores: need_scores,
            layer_boundary_hook: None,
            read_stage: None,
        });

        fwd_result?;

        self.read_logits(&self.logits_prefill_last)
    }

    fn step(&mut self, ctx: &StepCtx, token: u32) -> Result<Vec<f32>> {
        // 단일 토큰을 GPU decode_input 버퍼에 업로드한다.
        let bytes = token.to_ne_bytes();
        self.backend.write_buffer(&mut self.decode_input, &bytes)?;

        // AB-2 §5.7.1 (A안): persistent KIVI handle 의 Arc clone 으로 dyn fmts 구성 (transient
        // 댄스 제거). need_scores 는 layer-0 handle query.
        let need_scores = self
            .kivi_caches
            .first()
            .is_some_and(|h| h.with_cache_mut(|c| c.is_awqe_enabled()));
        let dyn_fmts: Vec<Arc<dyn KVCacheFormat>> = self
            .kivi_caches
            .iter()
            .map(|a| a.clone() as Arc<dyn KVCacheFormat>)
            .collect();

        let backend = self.backend.clone();
        let memory_ref: *const dyn Memory = self.memory.as_ref();
        // SAFETY: self.memory는 self가 살아있는 동안 유효하다.
        // raw pointer는 현재 stack frame 내에서만 역참조된다.
        let memory: &dyn Memory = unsafe { &*memory_ref };

        let fwd_result = self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &self.decode_input,
            start_pos: ctx.pos,
            fmts: &dyn_fmts,
            backend: &backend,
            memory,
            logits_out: &mut self.logits_decode,
            x_gen: Some(&mut self.decode_x_gen),
            workspace: Some(&mut self.decode_workspace),
            logits_last_only: false,
            score_accumulator: None,
            query_stats_accumulator: None,
            skip_config: None,
            importance_collector: None,
            cache_self_need_scores: need_scores,
            layer_boundary_hook: None,
            read_stage: None,
        });

        fwd_result?;

        self.read_logits(&self.logits_decode)
    }

    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }

    fn on_kv_prune(&mut self, _new_pos: usize) {
        // D4: Phase 4-5 KIVI는 eviction 미지원.
        // Phase 4-6+에서 KiviCache position sync 구현 예정.
    }

    fn reset_kv(&mut self) -> anyhow::Result<()> {
        // AB-2 §5.7.1 (A안): persistent handle 의 interior-mut seam 경유 (with_cache_mut).
        for h in &self.kivi_caches {
            h.with_cache_mut(|c| c.reset());
        }
        Ok(())
    }
}

fn workspace_config_for(model: &TransformerModel, max_seq_len: usize) -> WorkspaceConfig {
    let head_dim = model.config.head_dim;
    let kv_dim = model.config.num_key_value_heads * head_dim;
    WorkspaceConfig {
        batch_size: 1,
        dim: model.config.hidden_size,
        q_dim: model.config.num_attention_heads * head_dim,
        k_dim: kv_dim,
        v_dim: kv_dim,
        ffn_hidden: model.config.intermediate_size,
        n_heads: model.config.num_attention_heads,
        max_seq_len,
    }
}

fn alloc_logits(
    memory: &dyn Memory,
    backend: Arc<dyn Backend>,
    vocab_size: usize,
) -> Result<Tensor> {
    let buf = memory.alloc(vocab_size * 4, DType::F32)?;
    Ok(Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        buf,
        backend,
    ))
}

/// `num_layers`개의 [`KiviCache`]를 일괄 할당한다.
///
/// GPU 백엔드가 OpenCL일 때 `KiviCache::new_gpu`를 사용하고, 그 외에는
/// CPU 모드 `KiviCache::new_with_bits`를 사용한다.
///
/// `kivi` 는 KIVI native attention capability handle (Phase α-W-4 §3.3). caller 가
/// `caps.get::<dyn KiviAttentionBackend>()` 로 pull 해 전달한다. R3 불변식: OpenCL
/// backend 면 반드시 `Some` (init.rs 가 OpenCL 에 register).
#[allow(clippy::too_many_arguments)]
pub fn alloc_kivi_kv_caches(
    num_layers: usize,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    residual_size: usize,
    bits: u8,
    backend: &Arc<dyn Backend>,
    kivi: &Option<Arc<dyn KiviAttentionBackend>>,
    memory: &Arc<dyn Memory>,
) -> Vec<KiviCache> {
    if backend.name() == "OpenCL" {
        (0..num_layers)
            .map(|_| {
                KiviCache::new_gpu(
                    kv_heads,
                    head_dim,
                    max_seq_len,
                    residual_size,
                    bits,
                    backend.clone(),
                    kivi.clone(),
                    memory.clone(),
                )
            })
            .collect()
    } else {
        (0..num_layers)
            .map(|_| KiviCache::new_with_bits(kv_heads, head_dim, max_seq_len, residual_size, bits))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `alloc_kivi_kv_caches`가 요청한 수만큼 KiviCache를 반환하는지 검증한다.
    /// 실제 모델 없이 cache 생성 경로만 검증한다.
    #[test]
    fn test_alloc_kivi_kv_caches_count() {
        use crate::backend::cpu::CpuBackend;
        use crate::memory::galloc::Galloc;

        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let memory: Arc<dyn Memory> = Arc::new(Galloc::new());
        let kivi: Option<Arc<dyn KiviAttentionBackend>> = None;

        // residual_size와 head_dim은 QKKV(=32)의 배수여야 한다.
        let caches = alloc_kivi_kv_caches(
            4,   // num_layers
            8,   // kv_heads
            64,  // head_dim (32의 배수)
            128, // max_seq_len
            32,  // residual_size (QKKV=32의 배수)
            2,   // bits
            &backend, &kivi, &memory,
        );

        assert_eq!(
            caches.len(),
            4,
            "num_layers개의 KiviCache가 생성되어야 한다"
        );
    }

    /// 각 KiviCache의 bits 설정이 요청한 값과 일치하는지 검증한다.
    #[test]
    fn test_alloc_kivi_kv_caches_bits() {
        use crate::backend::cpu::CpuBackend;
        use crate::memory::galloc::Galloc;

        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let memory: Arc<dyn Memory> = Arc::new(Galloc::new());
        let kivi: Option<Arc<dyn KiviAttentionBackend>> = None;

        // residual_size=32(QKKV), head_dim=64(QKKV*2) 로 제약 충족.
        for &bits in &[2u8, 4, 8] {
            let caches = alloc_kivi_kv_caches(2, 4, 64, 128, 32, bits, &backend, &kivi, &memory);
            for cache in &caches {
                assert_eq!(
                    cache.bits(),
                    bits,
                    "bits={} 설정이 KiviCache에 반영되어야 한다",
                    bits
                );
            }
        }
    }
}
