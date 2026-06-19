//! `forward_prefill_fmt` — `forward_prefill` 의 KVCacheFormat trait-object fork (Phase α-K substep ①-b).
//!
//! 갈래 2 (Generic → trait object): `forward_gen_fmt`(decode)의 prefill 짝.
//!
//! **branch-by-abstraction, additive**: 기존 `forward_prefill<C: KVCacheOps>`(forward.rs:41)를 1바이트도
//! 안 건드린다. 본 fork 는 **prefill PrefillWorkspace 라이브 경로**(forward.rs:114-787 의 happy-path:
//! partition off / profile off)만 재현하고, dead/optional 블록(partition_ctx
//! 626-740 / profiler / fallback-path 789-1271)은 전부 생략한다. 두 지점만
//! 위임:
//!   - KV update 블록(forward.rs:182-249)  → `fmt.write_kv_batch(&k_rope, &ws.v, backend)` (C3 흡수)
//!   - attention dispatch(forward.rs:251-585) → `fmt.attention_into(...)` (multi-token prefill arm = C-1)
//!
//! **bit-identical 범위**: 공유 골격(norm/QKV matmul/RoPE/O-proj/FFN/residual)은 forward_prefill 의
//! PrefillWorkspace 라이브 arm 과 같은 backend 호출. attention 위임(`StandardFormat::attention_into`
//! seq_len>1 prefill arm)은 forward_prefill 의 attention 블록을 정확히 미러하므로 F16/Q4_0/F32 모두
//! bit-identical(decode 와 달리 prefill 은 forward_prefill 도 inline-NEON 아닌 flash 경로). 생략한
//! instrumentation(prof/op-trace)은 수치 무관. score accumulation 은 prefill 에서 미수행
//! (forward_prefill 의 `_need_scores` 동일) → `attention_into` 에 scores=None.
//!
//! 모델별 라이브 분기는 `forward_gen_fmt`(decode) 커버리지와 1:1: qkv_bias / Gemma3 q_norm·k_norm /
//! pre_ffn_norm·post_ffn_norm / rms_norm_add_unit / gelu_tanh / is_local_attn+window.

use super::*;
use crate::format::{AttnDims, KVCacheFormat};

/// `forward_prefill_fmt` 인자 — `forward_prefill` 의 `kv_cache: &mut C` 를 `fmt: &Arc<dyn KVCacheFormat>`
/// 로 교체하고, partition/variance/profiler 인자는 라이브 미진입이라 드롭. `pws` = PrefillWorkspace
/// (decode 의 LayerWorkspace 와 별개 — multi-token alloc).
pub(crate) struct ForwardPrefillFmtArgs<'a> {
    pub x: &'a mut Tensor,
    pub fmt: &'a Arc<dyn KVCacheFormat>,
    pub start_pos: usize,
    pub backend: &'a Arc<dyn Backend>,
    pub pws: &'a mut crate::layers::workspace::PrefillWorkspace,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub head_dim: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub dim: usize,
    pub skip_attn: bool,
    pub skip_mlp: bool,
    /// Gemma3: true → `x * (1 + w) / rms(x)`, false → Llama/Qwen2.
    pub rms_norm_add_unit: bool,
    /// Gemma3: true → GELU_tanh, false → SiLU.
    pub use_gelu_tanh: bool,
    /// Gemma3: 이 layer 가 local(SWA) attention 인가.
    pub is_local_attn: Option<bool>,
    /// Gemma3: local attention window.
    pub local_attn_window: Option<usize>,
}

impl TransformerLayer {
    /// `forward_prefill` 의 trait-object fork (prefill, seq_len>1). KV write + attention 만 fmt 위임.
    pub(crate) fn forward_prefill_fmt(&self, args: ForwardPrefillFmtArgs) -> Result<()> {
        // layer-skip: 두 sub-layer 모두 skip 이면 identity (forward_gen_fmt:59 동치).
        if args.skip_attn && args.skip_mlp {
            return Ok(());
        }

        let x = args.x;
        let fmt = args.fmt;
        let start_pos = args.start_pos;
        let backend = args.backend;
        let ws = args.pws;
        let rms_norm_eps = args.rms_norm_eps;
        let rope_theta = args.rope_theta;
        let rms_norm_add_unit = args.rms_norm_add_unit;
        let use_gelu_tanh = args.use_gelu_tanh;
        let head_dim = args.head_dim;
        let batch_size = args.batch_size;
        let seq_len = args.seq_len;
        let dim = args.dim;

        let q_dim = self.wq.shape().dims()[0];
        let k_dim = self.wk.shape().dims()[0];
        let v_dim = self.wv.shape().dims()[0];
        let n_heads_q = q_dim / head_dim;
        let n_heads_kv = k_dim / head_dim;

        // Reshape workspace tensors to actual seq_len (forward.rs:116-122).
        ws.residual
            .reshape(Shape::new(vec![batch_size, seq_len, dim]));
        ws.residual_ffn
            .reshape(Shape::new(vec![batch_size, seq_len, dim]));
        ws.q.reshape(Shape::new(vec![batch_size, seq_len, q_dim]));
        ws.k.reshape(Shape::new(vec![batch_size, seq_len, k_dim]));
        ws.v.reshape(Shape::new(vec![batch_size, seq_len, v_dim]));

        let n_elem = batch_size * seq_len * dim;

        // 1. Attention Norm — residual = x; x = norm(x) (forward.rs:126-131).
        backend.copy_slice(x, &mut ws.residual, 0, 0, n_elem)?;
        backend.rms_norm(x, &self.attention_norm, rms_norm_eps, rms_norm_add_unit)?;

        // 2. QKV projections (GPU-only; partition 제거됨).
        backend.matmul_transposed(x, &self.wq, &mut ws.q)?;
        backend.matmul_transposed(x, &self.wk, &mut ws.k)?;
        backend.matmul_transposed(x, &self.wv, &mut ws.v)?;

        // QKV bias (Qwen2 등).
        if let Some(ref bias) = self.qkv_bias {
            backend.add_row_bias(&mut ws.q, &bias.bq)?;
            backend.add_row_bias(&mut ws.k, &bias.bk)?;
            backend.add_row_bias(&mut ws.v, &bias.bv)?;
        }

        // Gemma3 QK-Norm.
        if let Some(ref q_norm_w) = self.q_norm {
            let total_q_heads = batch_size * seq_len * n_heads_q;
            let saved = ws.q.shape().clone();
            ws.q.reshape(Shape::new(vec![total_q_heads, head_dim]));
            backend.rms_norm(&mut ws.q, q_norm_w, rms_norm_eps, true)?;
            ws.q.reshape(saved);
        }
        if let Some(ref k_norm_w) = self.k_norm {
            let total_k_heads = batch_size * seq_len * n_heads_kv;
            let saved = ws.k.shape().clone();
            ws.k.reshape(Shape::new(vec![total_k_heads, head_dim]));
            backend.rms_norm(&mut ws.k, k_norm_w, rms_norm_eps, true)?;
            ws.k.reshape(saved);
        }

        // 3. RoPE (multi-token shape [batch, seq_len, n_heads, head_dim]).
        let mut q_rope = Tensor::new(
            Shape::new(vec![batch_size, seq_len, n_heads_q, head_dim]),
            ws.q.buffer().clone(),
            backend.clone(),
        );
        let mut k_rope = Tensor::new(
            Shape::new(vec![batch_size, seq_len, n_heads_kv, head_dim]),
            ws.k.buffer().clone(),
            backend.clone(),
        );
        backend.rope_inplace(&mut q_rope, start_pos, rope_theta)?;
        backend.rope_inplace(&mut k_rope, start_pos, rope_theta)?;

        // 4. KV cache update (multi-token) → fmt.write_kv_batch (C3).
        fmt.write_kv_batch(&k_rope, &ws.v, backend.as_ref())?;

        // 5. Attention (multi-token causal) → fmt.attention_into prefill arm (C-1).
        ws.out_attn
            .reshape(Shape::new(vec![batch_size, seq_len, q_dim]));
        let window = if matches!(args.is_local_attn, Some(true)) {
            args.local_attn_window
        } else {
            None
        };
        fmt.attention_into(
            &q_rope,
            backend.as_ref(),
            &mut ws.out_attn,
            AttnDims { n_heads_q, window },
            None,
        )?;

        // 6. Output projection.
        ws.attn_out_proj
            .reshape(Shape::new(vec![batch_size, seq_len, dim]));
        backend.matmul_transposed(&ws.out_attn, &self.wo, &mut ws.attn_out_proj)?;

        // Gemma3: post-attention norm on O-proj output.
        if rms_norm_add_unit {
            backend.rms_norm(&mut ws.attn_out_proj, &self.ffn_norm, rms_norm_eps, true)?;
        }
        // 7. Post-attention residual.
        backend.add_assign(&mut ws.attn_out_proj, &ws.residual)?;
        backend.copy_slice(&ws.attn_out_proj, x, 0, 0, n_elem)?;

        // 8. Pre-FFN norm (residual_ffn = x; x = norm(x)).
        backend.copy_slice(x, &mut ws.residual_ffn, 0, 0, n_elem)?;
        if let Some(ref pfn) = self.pre_ffn_norm {
            backend.rms_norm(x, pfn, rms_norm_eps, true)?;
        } else {
            backend.rms_norm(x, &self.ffn_norm, rms_norm_eps, false)?;
        }

        // 9. FFN gate + up (GPU-only; partition 제거됨).
        let ffn_hidden = self.w_up.shape().dims()[0];
        ws.gate
            .reshape(Shape::new(vec![batch_size, seq_len, ffn_hidden]));
        ws.up
            .reshape(Shape::new(vec![batch_size, seq_len, ffn_hidden]));
        ws.down.reshape(Shape::new(vec![batch_size, seq_len, dim]));

        backend.matmul_transposed(x, &self.w_gate, &mut ws.gate)?;
        backend.matmul_transposed(x, &self.w_up, &mut ws.up)?;
        if use_gelu_tanh {
            backend.gelu_tanh_mul(&mut ws.gate, &ws.up)?;
        } else {
            backend.silu_mul(&mut ws.gate, &ws.up)?;
        }
        backend.matmul_transposed(&ws.gate, &self.w_down, &mut ws.down)?;
        if let Some(ref pfn) = self.post_ffn_norm {
            backend.rms_norm(&mut ws.down, pfn, rms_norm_eps, true)?;
        }
        // 10. Residual 2.
        backend.add_assign(&mut ws.down, &ws.residual_ffn)?;
        backend.copy_slice(&ws.down, x, 0, 0, n_elem)?;

        Ok(())
    }
}
