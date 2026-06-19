//! `QuantWindowFormat` — `KVCacheFormat` impl wrapping a `QuantizedRecentWindowCache` (§4.1, Phase α-K).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §4.1 (R4 ④ quant-window creep 제거 + AWQE 자가 흡수).
//!
//! **purely additive wrapper, now LIVE** — 기존 `QuantizedRecentWindowCache`/`KVCacheOps` 를 1바이트도 건드리지
//! 않는 신규 wrapper 로 출발했으나, production quant-window forward 경로가 이제 이 wrapper 를 생성한다
//! (`session/forward/quant_window_forward.rs`). 내부 가변성 = `std::sync::Mutex`.
//!
//! `attention_into` 는 quant_attn-native(GPU fused dequant) 와 fallback(F32 view →
//! `backend.attention_gen`) 에 더해 AWQE 자가 흡수(scores `Some` 일 때 내부
//! `QuantizedRecentWindowCache.set_attn_scores` 로 자가 기록)를 수행한다. base trait 에 `needs_attn_scores`
//! 메서드를 만들지 않는다(§4.1 R4 ③).

use std::sync::Mutex;

use anyhow::Result;

use crate::backend::Backend;
use crate::format::{AttnDims, KVCacheFormat};
use crate::kv::quant_window_cache::QuantizedRecentWindowCache;
use crate::tensor::Tensor;

/// quant-window (Q2 + residual) KV cache 를 `KVCacheFormat` 으로 노출하는 wrapper.
///
/// 기존 `QuantizedRecentWindowCache` 를 `Mutex` 로 감싸 `&self` 메서드에서 내부 `&mut` 메서드에 위임한다.
/// `QuantizedRecentWindowCache` 자체는 무변.
pub struct QuantWindowFormat {
    idx: usize,
    inner: Mutex<QuantizedRecentWindowCache>,
}

impl QuantWindowFormat {
    /// `QuantizedRecentWindowCache` 를 layer 인덱스와 함께 wrapping. (quant-window forward 경로가 생성 — live.)
    pub fn new(idx: usize, inner: QuantizedRecentWindowCache) -> Self {
        Self {
            idx,
            inner: Mutex::new(inner),
        }
    }

    /// KV write 흡수 — `QuantizedRecentWindowCache` 는 CPU-only(`get_buffers_mut`==None) 라 GPU scatter fast-path
    /// 대상이 아니다. 구 `update_kv_cache`(transformer_layer.rs:31) 의 CPU-only 분기를 옮긴 것:
    /// producer tensor 가 host-mapped GPU 메모리(non-null ptr)면 device 커널 완료 전 stale read
    /// 방지를 위해 `synchronize` 후 `QuantizedRecentWindowCache::update`(Q2 quant + residual append 자체 수행) 호출.
    ///
    /// decode/prefill 동일 경로(`QuantizedRecentWindowCache::update` 가 seq_len 으로 분기). device-only producer
    /// (`as_ptr()` null)의 명시적 readback 은 후속 device substep 으로 연기(host 미발생).
    fn write_inner(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        let mut cache = self.inner.lock().unwrap();
        if !new_k.as_ptr().is_null() {
            backend.synchronize()?;
        }
        cache.update(new_k, new_v)
    }

    /// 내부 `QuantizedRecentWindowCache` 에 `&mut` 접근하여 `f` 실행 (AB-2 §5.7.1 — `StandardFormat::with_cache_mut`
    /// (standard_format.rs:65) verbatim 동형).
    ///
    /// `QuantWindowFormat` 은 이미 `Mutex<QuantizedRecentWindowCache>` interior-mutable 이므로 `&self` 로 transition_bits·
    /// reset 등 non-forward 연산에 도달하는 seam 을 concrete inherent 로 제공한다(base trait 무변).
    /// lock guard 안에서 closure 를 실행하므로 호출 종료 시 lock 이 풀린다.
    pub(crate) fn with_cache_mut<R>(
        &self,
        f: impl FnOnce(&mut QuantizedRecentWindowCache) -> R,
    ) -> R {
        let mut guard = self.inner.lock().unwrap();
        f(&mut guard)
    }

    /// 현재 양자화 bit-width (AB-2 §5.7.6 — heartbeat kv_dtype query 용).
    ///
    /// `QuantizedRecentWindowCache::bits()`(quant_window_cache.rs:406) 위임. ResilienceAdapter 가 layer-0 QuantWindowFormat handle
    /// 에서 현재 bits 를 query 해 `bits→dtype` 문자열로 매핑한다.
    pub(crate) fn current_bits(&self) -> u8 {
        self.inner.lock().unwrap().bits()
    }

    /// wrapping 을 해제하고 내부 `QuantizedRecentWindowCache` 를 반환 (Phase α-K ①-c eval transient-wrap round-trip).
    ///
    /// `StandardFormat::into_inner` 대칭. eval 이 forward 1회 동안만 `Vec<QuantizedRecentWindowCache>` →
    /// `Arc<QuantWindowFormat>` 로 wrap 후 `Arc::try_unwrap().into_inner()` 로 복귀시킨다. base trait 무변.
    pub(crate) fn into_inner(self) -> QuantizedRecentWindowCache {
        self.inner.into_inner().unwrap()
    }
}

impl crate::session::resilience_adapter::QuantStageHandle for QuantWindowFormat {
    /// §4.5: heartbeat kv_dtype query — `current_bits()` 위임. base trait 무변(중립 sub-trait).
    fn current_kv_bits(&self) -> u8 {
        self.current_bits()
    }
}

impl KVCacheFormat for QuantWindowFormat {
    fn idx(&self) -> usize {
        self.idx
    }

    fn current_pos(&self) -> usize {
        self.inner.lock().unwrap().current_pos()
    }

    fn capacity(&self) -> usize {
        self.inner.lock().unwrap().capacity()
    }

    fn write_kv(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        self.write_inner(new_k, new_v, backend)
    }

    fn write_kv_batch(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        self.write_inner(new_k, new_v, backend)
    }

    fn attention_into(
        &self,
        q: &Tensor,
        backend: &dyn Backend,
        out: &mut Tensor,
        dims: AttnDims,
        scores: Option<&mut [f32]>,
    ) -> Result<()> {
        let mut cache = self.inner.lock().unwrap();
        let n_heads_q = dims.n_heads_q;

        // ── prefill (seq_len>1): multi-token causal attention (Phase α-K ①-e) ──
        // quant-window 는 multi-token prefill native 커널 부재(attention_gen / attention_native 는 single-query
        // decode 전용 — causal-mask 없음)라, dequantized view(get_view) + StandardFormat 의
        // `prefill_attention`(free fn, pub(crate)) 재사용으로 처리한다(DRY). OLD generic
        // `forward_prefill<C>`(forward.rs:251-585)의 quant-window 경로(get_view → flash_attention_prefill /
        // flash_attention_forward_strided)와 bit-identical: quant-window CPU(SeqMajor F32) / GPU(bits=16
        // HeadMajor, bits 2/4/8 assembled) 모두 `kv_layout`/`kv_capacity` 인자로 분기된다.
        // `q_start_pos = cache_seq_len - seq_len`(= forward_prefill 의 start_pos, write 후 불변식).
        // prefill 은 score 누적 안 함(forward_prefill 의 `_need_scores` 동일) → `scores` 무시.
        let seq_len = q.shape().dims()[1];
        if seq_len > 1 {
            let n_heads_kv = cache.kv_heads();
            let head_dim = cache.head_dim();
            let kv_capacity = cache.capacity();
            let kv_layout = cache.layout();
            let cache_seq_len = cache.current_pos();
            let batch_size = q.shape().dims()[0];
            let q_start_pos = cache_seq_len - seq_len;
            let (k_cache, v_cache) = cache.get_view();
            let _ = scores;
            return crate::kv::standard_format::prefill_attention(
                q,
                out,
                &k_cache,
                &v_cache,
                n_heads_q,
                n_heads_kv,
                head_dim,
                seq_len,
                cache_seq_len,
                kv_capacity,
                batch_size,
                kv_layout,
                q_start_pos,
                dims.window,
                backend,
            );
        }

        // quant_attn-native 경로 게이팅(host 미검증 — device 검증은 후속 substep). 게이팅 조건만 미리
        // 평가하고(borrow 분리), dispatch 는 별도 헬퍼에 위임해 scores ownership 을 단일 경로로 가둔다.
        // get_quant_window_raw_buffers 가 Some + backend 가 QuantAttnBackend + has_quant_attn_kernel +
        // is_nosub_device(NVIDIA) + 토큰 존재 일 때만. Adreno(subgroup)는 F32 dequant 경로가 더
        // 빠르므로 native 미사용(forward_gen 의 기존 게이팅 보존).
        // Stage E: the cap is pulled from the cache's own `quant_attn` handle (the
        // `caps.get` Arc — the dlopen plugin or, pre-Stage-E, the same OpenCLBackend
        // object) rather than `backend.as_quant_attn()`, and the nosub device
        // property is read off the backend directly (byte-identical to the prior
        // `cap.is_nosub_device()` which delegated to `OpenCLBackend::is_nosub`).
        let use_native = backend.is_gpu()
            && backend.is_nosub_device()
            && cache
                .get_quant_window_raw_buffers()
                .zip(cache.quant_attn_cap())
                .map(|(raw, quant_attn_be)| {
                    quant_attn_be.has_quant_attn_kernel(raw.bits)
                        && (raw.q_tokens + raw.res_tokens) > 0
                })
                .unwrap_or(false);

        if use_native {
            return self.attention_native(q, backend, out, n_heads_q, scores, &mut cache);
        }

        // fallback: dequantized F32 view → backend.attention_gen (CPU-testable).
        let n_heads_kv = cache.kv_heads();
        let head_dim = cache.head_dim();
        let (k_cache, v_cache) = cache.get_view();
        let cache_seq_len = cache.current_pos();
        let effective_cache_len = match dims.window {
            Some(w) => cache_seq_len.min(w),
            None => cache_seq_len,
        };

        // attention_gen 에 caller scores 슬라이스를 직접 넘긴 뒤 AWQE 자가 흡수
        // (set_attn_scores 는 awqe_enabled=false 면 자체 no-op, §4.1 R4 ③).
        match scores {
            Some(s) => {
                backend.attention_gen(
                    q,
                    &k_cache,
                    &v_cache,
                    out,
                    n_heads_q,
                    n_heads_kv,
                    head_dim,
                    effective_cache_len,
                    Some(s),
                )?;
                let stride = s.len().checked_div(n_heads_q).unwrap_or(0);
                cache.set_attn_scores(s, n_heads_q, stride, effective_cache_len);
            }
            None => {
                backend.attention_gen(
                    q,
                    &k_cache,
                    &v_cache,
                    out,
                    n_heads_q,
                    n_heads_kv,
                    head_dim,
                    effective_cache_len,
                    None,
                )?;
            }
        }
        Ok(())
    }
}

impl QuantWindowFormat {
    /// quant_attn-native GPU fused dequant+attention dispatch + AWQE 자가 흡수 (§4.1 R4 ④).
    ///
    /// host 미검증(컴파일만) — device 검증은 후속 wiring substep. `scores` 가 `Some` 이면 native
    /// 커널이 임시 버퍼에 쓴 raw post-softmax score 를 caller 슬라이스로 복사 + 내부
    /// `QuantizedRecentWindowCache::set_attn_scores`(awqe_enabled 게이트 자가 처리)로 흡수한다.
    fn attention_native(
        &self,
        q: &Tensor,
        backend: &dyn Backend,
        out: &mut Tensor,
        n_heads_q: usize,
        scores: Option<&mut [f32]>,
        cache: &mut QuantizedRecentWindowCache,
    ) -> Result<()> {
        let n_heads_kv = cache.kv_heads();
        let head_dim = cache.head_dim();
        // Stage E: pull the cap from the cache's `quant_attn` handle (the same Arc
        // the gate checks), cloned so its borrow of `cache` ends before the
        // `set_attn_scores` (&mut) below. `backend` now only lends its cl_queue.
        let quant_attn_be = cache
            .quant_attn_cap()
            .expect("attention_native gated on quant_attn_cap().is_some()")
            .clone();
        let raw = cache
            .get_quant_window_raw_buffers()
            .expect("attention_native gated on get_quant_window_raw_buffers().is_some()");
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = raw.q_tokens + raw.res_tokens;

        // native 커널 score 임시 버퍼 (caller scores 유무로 게이팅).
        let mut tmp_scores: Vec<f32> = if scores.is_some() {
            vec![0.0; n_heads_q * total]
        } else {
            Vec::new()
        };
        // D8: ABI struct(cl_mem) 시그니처. `&Tensor` 6개를 raw cl_mem 으로 추출해
        // `QuantAttnArgs` 패킹. score 는 `(ptr, len)` 으로 변환(None → (null, 0)).
        // cl_queue 는 엔진의 live `cl_command_queue` 를 넘긴다(Stage E): borrowed-context
        // dlopen plugin 이 같은 in-order 큐에 enqueue 해야 score readback 순서가 보존된다.
        // 엔진 내장 OpenCL impl 은 이 슬롯을 무시하고 `&self.queue` 를 직접 쓰므로 무영향.
        use crate::backend::opencl::get_cl_mem;
        // `Mem::as_ptr()` 는 이미 `cl_mem`(= `*mut c_void`) 를 반환하므로 캐스트 불요.
        let q_mem = get_cl_mem(q.buffer().as_ref())?.as_ptr();
        let qk_mem = get_cl_mem(raw.qk_buf.buffer().as_ref())?.as_ptr();
        let qv_mem = get_cl_mem(raw.qv_buf.buffer().as_ref())?.as_ptr();
        let res_k_mem = get_cl_mem(raw.res_k.buffer().as_ref())?.as_ptr();
        let res_v_mem = get_cl_mem(raw.res_v.buffer().as_ref())?.as_ptr();
        let out_mem = get_cl_mem(out.buffer().as_ref())?.as_ptr();
        let (scores_ptr, scores_len): (*mut f32, usize) = if scores.is_some() {
            (tmp_scores.as_mut_ptr(), tmp_scores.len())
        } else {
            (std::ptr::null_mut(), 0)
        };
        let args = crate::backend::QuantAttnArgs {
            cl_queue: backend.cl_command_queue_ptr(),
            q_mem,
            qk_mem,
            qv_mem,
            res_k_mem,
            res_v_mem,
            out_mem,
            scores_out: scores_ptr,
            scores_len,
            num_heads_q: n_heads_q,
            num_heads_kv: n_heads_kv,
            head_dim,
            q_tokens: raw.q_tokens,
            res_tokens: raw.res_tokens,
            res_cap: raw.res_cap,
            scale,
            bits: raw.bits,
        };
        let rc = quant_attn_be.attention_gen_quant(&args);
        if rc != 0 {
            anyhow::bail!("quant-window attention_gen_quant failed (rc={rc})");
        }
        // 이후 `raw`(cache immutable borrow) 미사용 → NLL 이 set_attn_scores(가변) 전에 borrow 종료.

        if let Some(dst) = scores {
            let n = tmp_scores.len().min(dst.len());
            dst[..n].copy_from_slice(&tmp_scores[..n]);
            let valid_len = tmp_scores.len().checked_div(n_heads_q).unwrap_or(0);
            cache.set_attn_scores(&tmp_scores, n_heads_q, valid_len, valid_len);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn f32_tensor(dims: Vec<usize>, data: &[f32]) -> Tensor {
        use crate::buffer::DType;
        let buf = Arc::new(SharedBuffer::new(data.len() * 4, DType::F32));
        let mut t = Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()));
        t.as_mut_slice::<f32>().copy_from_slice(data);
        t
    }

    // QuantizedRecentWindowCache 제약: residual_size 와 head_dim 모두 QKKV(=32) 의 배수여야 한다 (quant_window_cache.rs:333).
    const HD: usize = 32; // head_dim
    const RES: usize = 32; // residual_size
    const MAXSEQ: usize = 256;

    #[test]
    fn test_geometry_delegates_to_kivicache() {
        // QuantizedRecentWindowCache CPU mode (bits=2 default).
        let cache = QuantizedRecentWindowCache::new(2, HD, MAXSEQ, RES);
        let fmt = QuantWindowFormat::new(5, cache);
        assert_eq!(fmt.idx(), 5);
        assert_eq!(fmt.current_pos(), 0);
        // CPU mode capacity == max_seq_len.
        assert_eq!(fmt.capacity(), MAXSEQ);
    }

    #[test]
    fn test_write_kv_advances_pos() {
        let kv_heads = 2;
        let fmt = QuantWindowFormat::new(
            0,
            QuantizedRecentWindowCache::new(kv_heads, HD, MAXSEQ, RES),
        );

        let token = vec![1.0f32; kv_heads * HD];
        let k = f32_tensor(vec![1, 1, kv_heads, HD], &token);
        let v = f32_tensor(vec![1, 1, kv_heads, HD], &token);
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 1);

        let batch = vec![2.0f32; 2 * kv_heads * HD];
        let kb = f32_tensor(vec![1, 2, kv_heads, HD], &batch);
        let vb = f32_tensor(vec![1, 2, kv_heads, HD], &batch);
        fmt.write_kv_batch(&kb, &vb, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 3);
    }

    #[test]
    fn test_attention_into_cpu_fallback() {
        // CPU QuantizedRecentWindowCache (bits=2): write 2 tokens, run attention via F32 view fallback.
        // Q2 quantization introduces error, so we only assert the output is finite and
        // bounded by the (positive) V magnitude — the CPU-testable seam is "runs + produces
        // a sane attention output", not bit-exact dequant (that is quant-window's own concern).
        let kv_heads = 1;
        let n_heads_q = 1;
        let fmt = QuantWindowFormat::new(
            0,
            QuantizedRecentWindowCache::new(kv_heads, HD, MAXSEQ, RES),
        );

        let v_row = vec![3.0f32; HD];
        let k_row = vec![0.0f32; HD]; // zero K → uniform softmax
        for _ in 0..2 {
            let k = f32_tensor(vec![1, 1, kv_heads, HD], &k_row);
            let v = f32_tensor(vec![1, 1, kv_heads, HD], &v_row);
            fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }
        assert_eq!(fmt.current_pos(), 2);

        let q = f32_tensor(vec![1, 1, n_heads_q, HD], &[1.0; HD]);
        let mut out = f32_tensor(vec![1, 1, n_heads_q, HD], &[0.0; HD]);
        let backend = CpuBackend::new();
        let mut scores = vec![0.0f32; n_heads_q * 2];

        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: None,
            },
            Some(&mut scores),
        )
        .unwrap();

        // Output: uniform attention over (Q2-dequantized) identical V rows → finite, ≈ V.
        let o = out.as_slice::<f32>();
        for &x in o {
            assert!(x.is_finite(), "attention output must be finite, got {x}");
            assert!(
                (0.0..=6.0).contains(&x),
                "out should be bounded near V=3, got {x}"
            );
        }
        // post-softmax scores recorded into caller slice: 2 ~equal weights summing to ~1.
        let s: f32 = scores.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-3,
            "post-softmax weights sum to 1, got {s}"
        );
    }

    #[test]
    fn test_attention_into_prefill_causal_uniform() {
        // Phase α-K ①-e: multi-token prefill arm (seq_len>1). seq=4 < res_cap(=RES=32)라 Q2 flush
        // 미발생 → residual 이 raw F32 그대로 dequant(exact)되어 bit-exact 검증 가능. K=0 → 모든
        // score 0 → uniform softmax. V[pos]=pos(broadcast). causal mask 로 query row r 은 cache pos
        // 0..=r 만 attend → out[r] = mean(0..=r) = r/2. write_kv_batch(prefill write) +
        // attention_into(신규 prefill arm) 합동 검증 + causal-mask 확인(arm 부재 시 panic 회귀 가드).
        let kv_heads = 1;
        let n_heads_q = 1;
        let seq = 4;
        let fmt = QuantWindowFormat::new(
            0,
            QuantizedRecentWindowCache::new(kv_heads, HD, MAXSEQ, RES),
        );
        let backend = CpuBackend::new();

        let k_data = vec![0.0f32; seq * kv_heads * HD];
        let mut v_data = vec![0.0f32; seq * kv_heads * HD];
        for p in 0..seq {
            for d in 0..HD {
                v_data[p * kv_heads * HD + d] = p as f32;
            }
        }
        let kb = f32_tensor(vec![1, seq, kv_heads, HD], &k_data);
        let vb = f32_tensor(vec![1, seq, kv_heads, HD], &v_data);
        fmt.write_kv_batch(&kb, &vb, &backend).unwrap();
        assert_eq!(fmt.current_pos(), seq);

        // q 값은 무관(K=0 → score 0). out = [1, seq, n_heads_q*head_dim].
        let q = f32_tensor(
            vec![1, seq, n_heads_q, HD],
            &vec![1.0; seq * n_heads_q * HD],
        );
        let mut out = f32_tensor(
            vec![1, seq, n_heads_q * HD],
            &vec![0.0; seq * n_heads_q * HD],
        );

        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
        )
        .unwrap();

        let o = out.as_slice::<f32>();
        for r in 0..seq {
            let expected = r as f32 / 2.0; // mean(0..=r), causal mask
            for d in 0..HD {
                let got = o[r * HD + d];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "row {r} d {d}: expected {expected} (causal mean 0..=r), got {got}"
                );
            }
        }
    }

    #[test]
    fn test_attention_into_no_scores() {
        // scores=None path must not panic and must produce output.
        let kv_heads = 1;
        let n_heads_q = 1;
        let fmt = QuantWindowFormat::new(
            0,
            QuantizedRecentWindowCache::new(kv_heads, HD, MAXSEQ, RES),
        );
        let row = vec![1.0f32; HD];
        let k = f32_tensor(vec![1, 1, kv_heads, HD], &row);
        let v = f32_tensor(vec![1, 1, kv_heads, HD], &row);
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();

        let q = f32_tensor(vec![1, 1, n_heads_q, HD], &row);
        let mut out = f32_tensor(vec![1, 1, n_heads_q, HD], &[0.0; HD]);
        let backend = CpuBackend::new();
        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
        )
        .unwrap();
        let o = out.as_slice::<f32>();
        for &x in o {
            assert!(x.is_finite());
        }
    }
}
