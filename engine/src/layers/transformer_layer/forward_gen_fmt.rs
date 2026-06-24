//! `forward_gen_fmt` — `forward_gen` 의 KVCacheFormat trait-object fork (Phase α-K substep 3c).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §9.1 (3c) — decode fallback 의 KV write +
//! attention 을 `Arc<dyn KVCacheFormat>` 로 flip. (갈래 2: Generic → trait object).
//!
//! **branch-by-abstraction, additive**: 기존 `forward_gen<C: KVCacheOps>` 를 1바이트도 안 건드린다.
//! 본 fork 는 **decode 라이브 경로**(partition off, fused off)만 정확히 재현하고, dead branch(partition
//! / fused QKV·FFN / kivi_native / F32 batch-scatter / CPU inline NEON attention)는 전부 생략한다
//! (census 확정 — 게이트에서 미진입). 두 지점만 위임:
//!   - KV update 블록(forward_gen.rs:332-386)  → `fmt.write_kv(&k_rope, &ws.v, backend)` (3a/3b 흡수)
//!   - attention dispatch(forward_gen.rs:463-1068) → `fmt.attention_into(...)` (Q4-GPU-fallback 포함)
//!
//! **bit-identical 범위 (중요)**: 공유 골격(norm/QKV matmul/RoPE/O-proj/FFN/residual)은 forward_gen 의
//! 라이브 arm 과 같은 backend 호출. attention 위임은 **F16 KV / Q4_0 KV / F32-device-only(null host ptr)
//! 에서만 bit-identical** — 이 셋은 forward_gen 도 `backend.attention_gen`(또는 Q4 fallback)로 가기
//! 때문. **⚠️ F32 KV + host-mapped 버퍼**(`--opencl-rpcmem` non-null / `--zero-copy` mapped / CPU
//! backend)는 forward_gen 이 inline-NEON attention(forward_gen.rs:554-1068, kv_start_pos 적용)을 타는
//! 반면 `attention_into` 는 무조건 `backend.attention_gen` 위임 → FP 누산 순서 상이 **NOT bit-identical**.
//! 따라서 (3c) device 게이트는 **F16/Q4_0 KV**(default=F16) 또는 F32-device-only 만 대상으로 한다.
//! 생략한 instrumentation(prof/op-trace/set_label)은 수치 무관(게이트는 `--profile`·env 미사용).
//! `set_attn_scores`/`needs_attn_scores` OR 항 생략은 StandardKVCache trait default(no-op/false)라 안전
//! (quant-window 전용). score_offset/effective_cache_len 은 forward_gen.rs:404 와 동일 식으로 재현. GPU score
//! acc layer_idx routing 도 동일 위치 보존.

use super::*;
use crate::format::{AttnDims, KVCacheFormat};

/// decode read-plan 라우팅. 활성 read stage 가 어디서 `read_plan` 을 산출했는지 구분한다.
///
/// `ForwardGenFmtArgs::read_routing == None`(production 기본)이면 full read(byte-identical).
pub(crate) enum ReadRouting<'a> {
    /// 호출 측이 이미 `read_plan` 을 산출·검증해 `select` 를 넘긴다. offload decode 가 layer i 의 plan 을
    /// layer i+1 prefetch 힌트로 써야 해서 `forward_gen_fmt` *이전*에 read_plan 을 돌리는 경로 — query 가
    /// 아직 없어 **query-agnostic proxy(또는 QueryStats)** 로 선택된 plan 이다.
    Precomputed {
        select: &'a [usize],
        granularity: argus_extension_api::ReadGranularity,
    },
    /// faithful: `forward_gen_fmt` 가 이 step·이 layer 의 **RoPE-적용 현재 Q** 를 GQA 환원해
    /// `read_plan` 을 내부에서(KV write *이전*, 기존 seam 과 동일한 캐시 뷰로) 호출한다. Quest 정본의
    /// current-Q 선택을 충실히 실행한다. (QueryStats running-mean 은 faithful Q 로 대체되어 더는 공급되지
    /// 않는다 — read_plan 에 query_stats=None 전달; QueryStats seam 은 dormant.)
    Faithful {
        stage: &'a dyn argus_extension_api::KVReadStage,
    },
}

/// `forward_gen_fmt` 인자 — `ForwardGenArgs` 의 `kv_cache: &mut C` 만 `fmt: &Arc<dyn KVCacheFormat>`
/// 로 교체. profiler/memory/is_last_layer 는 fmt 경로에서 불요(instrumentation·partition_fused 전용)
/// 라 드롭.
pub(crate) struct ForwardGenFmtArgs<'a> {
    pub x: &'a mut Tensor,
    pub fmt: &'a Arc<dyn KVCacheFormat>,
    pub start_pos: usize,
    pub backend: &'a Arc<dyn Backend>,
    pub ws: &'a mut crate::layers::workspace::LayerWorkspace,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// heavy-hitter/weighted-merge score 누적이 필요하면 attention_into 에 scores 버퍼 전달.
    pub need_scores: bool,
    pub head_dim: usize,
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
    pub layer_idx: usize,
    /// KV read-plan 라우팅. `Some(..)` 면 활성 format 의 `SelectiveRead::attention_into_selected` 로
    /// 선택적 읽기, `None`(production 기본)이면 기존 `attention_into`(full read). 분기 1회 외 happy path
    /// 비용 0(INV-147 동형). [`ReadRouting::Faithful`] 은 현재 Q 를 host 로 1회 읽어(opt-in) read_plan 을
    /// 내부 산출, [`ReadRouting::Precomputed`] 는 호출 측이 넘긴 select 를 그대로 쓴다.
    pub read_routing: Option<ReadRouting<'a>>,
}

impl TransformerLayer {
    /// `forward_gen` 의 trait-object fork (decode, seq_len=1). KV write + attention 만 fmt 위임.
    pub(crate) fn forward_gen_fmt(&self, args: ForwardGenFmtArgs) -> Result<()> {
        // layer-skip: 두 sub-layer 모두 skip 이면 identity (forward_gen.rs:26 동치).
        if args.skip_attn && args.skip_mlp {
            return Ok(());
        }

        let x = args.x;
        let fmt = args.fmt;
        let start_pos = args.start_pos;
        let backend = args.backend;
        let ws = args.ws;
        let rms_norm_eps = args.rms_norm_eps;
        let rope_theta = args.rope_theta;
        let rms_norm_add_unit = args.rms_norm_add_unit;
        let use_gelu_tanh = args.use_gelu_tanh;
        let head_dim = args.head_dim;
        let layer_idx = args.layer_idx;
        let batch_size = x.shape().dims()[0];
        let is_gpu = backend.is_gpu();

        // 1. Attention Norm — out-of-place: ws.residual = norm(x), x 보존(skip connection).
        backend.rms_norm_oop(
            x,
            &mut ws.residual,
            &self.attention_norm,
            rms_norm_eps,
            rms_norm_add_unit,
        )?;

        // 2. QKV projections (decode GPU 라이브 = matmul_transposed×3; fused QKV 는 aarch64 dead).
        crate::thread_pool::get_pool().begin_batch();
        backend.matmul_transposed(&ws.residual, &self.wq, &mut ws.q)?;
        backend.matmul_transposed(&ws.residual, &self.wk, &mut ws.k)?;
        backend.matmul_transposed(&ws.residual, &self.wv, &mut ws.v)?;
        crate::thread_pool::get_pool().end_batch();
        if is_gpu && std::env::var_os("LLMRS_DISABLE_FLUSH_QKV").is_none() {
            backend.flush()?;
        }

        // QKV bias (Qwen2 등).
        if let Some(ref bias) = self.qkv_bias {
            backend.add_row_bias(&mut ws.q, &bias.bq)?;
            backend.add_row_bias(&mut ws.k, &bias.bk)?;
            backend.add_row_bias(&mut ws.v, &bias.bv)?;
        }

        // 3. RoPE.
        let q_dim = self.wq.shape().dims()[0];
        let k_dim = self.wk.shape().dims()[0];
        let n_heads_q = q_dim / head_dim;
        let n_heads_kv = k_dim / head_dim;

        // Gemma3 QK-Norm: per-head RMSNorm on Q/K before RoPE.
        if let Some(ref q_norm_w) = self.q_norm {
            let total_q_heads = batch_size * n_heads_q;
            let saved_shape = ws.q.shape().clone();
            ws.q.reshape(Shape::new(vec![total_q_heads, head_dim]));
            backend.rms_norm(&mut ws.q, q_norm_w, rms_norm_eps, true)?;
            ws.q.reshape(saved_shape);
        }
        if let Some(ref k_norm_w) = self.k_norm {
            let total_k_heads = batch_size * n_heads_kv;
            let saved_shape = ws.k.shape().clone();
            ws.k.reshape(Shape::new(vec![total_k_heads, head_dim]));
            backend.rms_norm(&mut ws.k, k_norm_w, rms_norm_eps, true)?;
            ws.k.reshape(saved_shape);
        }

        let mut q_rope = Tensor::new(
            Shape::new(vec![batch_size, 1, n_heads_q, head_dim]),
            ws.q.buffer().clone(),
            backend.clone(),
        );
        let mut k_rope = Tensor::new(
            Shape::new(vec![batch_size, 1, n_heads_kv, head_dim]),
            ws.k.buffer().clone(),
            backend.clone(),
        );
        backend.rope_inplace(&mut q_rope, start_pos, rope_theta)?;
        backend.rope_inplace(&mut k_rope, start_pos, rope_theta)?;

        // 3.5 faithful read-plan seam (Quest 정본 current-Q). KV write *이전*에 산출 → read_plan 이 보는
        // 캐시 뷰(current_pos=P, 현재 토큰 미반영)와 검증 상한이 기존 seam(transformer.rs:1650)과 동일.
        // 바뀌는 것은 query 신호뿐: running-mean 근사 → 이 step·layer 의 RoPE-적용 현재 Q. read_routing 이
        // Faithful 이고 format 이 SelectiveRead 면, 현재 Q 를 host 로 1회 읽어 GQA 환원([n_kv_heads*head_dim])
        // 한 뒤 read_plan→검증해 owned select 를 만든다. production(read_routing=None)은 미진입(비용 0,
        // byte-identical).
        //
        // ★호스트 읽기 = `synchronize()`(GPU=clFinish, CPU=no-op) 로 RoPE 커널 완료를 보장한 뒤
        // `read_buffer` 로 현재 Q 를 host Vec 으로 복사한다(= `read_logits`/standard_format INV-191 과 동일한
        // 라이브 GPU→host 패턴). ⚠ decode workspace 의 `ws.q`/`q_rope` 버퍼는 Adreno 에서 device-only
        // (host ptr=null) → `as_slice` 직접 읽기는 null deref(SEGV) → 반드시 read_buffer 경유(INV-191).
        let faithful_select: Option<(Vec<usize>, argus_extension_api::ReadGranularity)> =
            if let Some(ReadRouting::Faithful { stage }) = args.read_routing.as_ref() {
                fmt.as_selective_read()
                    .and_then(|sr| {
                        // RoPE 커널 완료 보장 후 device→host 복사. sync/read 실패 시 None → full read 폴백
                        // (plan 은 best-effort 힌트, 정확성 계약 아님).
                        backend.synchronize().ok()?;
                        let mut q_host = vec![0.0f32; q_rope.size() / 4];
                        // SAFETY: q_host 는 막 할당한 f32 슬라이스 — [u8] 재해석은 read_buffer 가 GPU 버퍼
                        // 바이트를 host 로 쓰는 데 안전하고, backend 는 호출 후 포인터를 보유하지 않는다.
                        let q_bytes = unsafe {
                            std::slice::from_raw_parts_mut(
                                q_host.as_mut_ptr() as *mut u8,
                                q_host.len() * 4,
                            )
                        };
                        backend.read_buffer(&q_rope, q_bytes).ok()?;
                        let reduced = gqa_reduce_query(&q_host, n_heads_q, n_heads_kv, head_dim);
                        // query_stats=None: faithful current-Q 가 running-mean 을 대체(QueryStats dormant).
                        sr.read_plan(*stage, layer_idx, Some(&reduced), None)
                    })
                    .and_then(|plan| {
                        crate::models::transformer::TransformerModel::validate_read_plan(
                            &plan,
                            fmt.current_pos(),
                        )
                        .map(|sel| (sel.to_vec(), plan.granularity))
                    })
            } else {
                None
            };

        // 4. KV cache update → fmt.write_kv (3a/3b 흡수: GPU F16/F32 scatter / 비-F32 cast / F32 update).
        fmt.write_kv(&k_rope, &ws.v, backend.as_ref())?;

        // 5. Attention → fmt.attention_into (Q4-GPU-fallback 포함, 2단계에서 흡수).
        let cache_seq_len = fmt.current_pos();
        // Sliding window 는 is_local_attn==Some(true) 일 때만 적용 (forward_gen.rs:397 게이팅 동치).
        let window = if matches!(args.is_local_attn, Some(true)) {
            args.local_attn_window
        } else {
            None
        };
        let effective_cache_len = match window {
            Some(w) => cache_seq_len.min(w),
            None => cache_seq_len,
        };
        // score accumulator 의 ws.scores[t] → cache pos 매핑용 offset (forward_gen.rs:404 동일 식).
        ws.score_offset = cache_seq_len - effective_cache_len;

        // StandardKVCache 는 needs_attn_scores()=false (trait default) 라 OR 항 불요.
        let need_scores = args.need_scores;

        // GPU score acc layer_idx routing (forward_gen.rs:505-509) — base trait 불변, backend 핸들 경유.
        if let Some(gpu_acc) = backend.gpu_score_acc_mut()
            && gpu_acc.is_active()
        {
            gpu_acc.set_current_layer_idx(layer_idx);
        }

        let scores_arg = if need_scores {
            Some(&mut ws.scores[..])
        } else {
            None
        };
        // read-plan 라우팅. Precomputed(offload, 호출 측 산출 select) 또는 Faithful(위 3.5 에서 현재 Q 로
        // 산출한 owned select). 활성 format 이 SelectiveRead capability 를 노출하면 선택적 읽기, 아니면
        // (미지원 format) plan 무시 + full read 폴백(D4). happy path(read_routing=None)는 분기 1회 —
        // full read 직행(INV-147 byte-identical).
        let read_select: Option<(&[usize], argus_extension_api::ReadGranularity)> =
            match args.read_routing.as_ref() {
                Some(ReadRouting::Precomputed {
                    select,
                    granularity,
                }) => Some((select, *granularity)),
                _ => faithful_select.as_ref().map(|(s, g)| (s.as_slice(), *g)),
            };
        match read_select {
            Some((select, granularity)) if fmt.as_selective_read().is_some() => {
                let sr = fmt
                    .as_selective_read()
                    .expect("as_selective_read().is_some() checked just above");
                sr.attention_into_selected(
                    &q_rope,
                    backend.as_ref(),
                    &mut ws.out_attn,
                    AttnDims { n_heads_q, window },
                    select,
                    granularity,
                    scores_arg,
                )?;
            }
            _ => {
                fmt.attention_into(
                    &q_rope,
                    backend.as_ref(),
                    &mut ws.out_attn,
                    AttnDims { n_heads_q, window },
                    scores_arg,
                    None, // R-P1-1: decode 는 PFA 미산출.
                )?;
            }
        }
        // set_attn_scores(forward_gen.rs:1071) 는 StandardKVCache no-op(quant-window AWQE 전용) → 생략.

        // 6. Output projection.
        backend.matmul_transposed(&ws.out_attn, &self.wo, &mut ws.attn_out)?;

        // 7+8. Post-attention residual + pre-FFN norm.
        if rms_norm_add_unit {
            // Gemma3: post-attn norm(ffn_norm) → fused add + pre_ffn_norm.
            backend.rms_norm(&mut ws.attn_out, &self.ffn_norm, rms_norm_eps, true)?;
            if let Some(ref pfn) = self.pre_ffn_norm {
                backend.add_rms_norm_oop(
                    x,
                    &ws.attn_out,
                    &mut ws.residual,
                    pfn,
                    rms_norm_eps,
                    true,
                )?;
            } else {
                backend.add_assign(x, &ws.attn_out)?;
                backend.copy_into(x, &mut ws.residual)?;
            }
        } else {
            // Llama/Qwen2: fused add + norm.
            backend.add_rms_norm_oop(
                x,
                &ws.attn_out,
                &mut ws.residual,
                &self.ffn_norm,
                rms_norm_eps,
                false,
            )?;
        }

        // 9. FFN gate + up (decode GPU 라이브 = else arm; fused NEON 은 dead).
        crate::thread_pool::get_pool().begin_batch();
        if !use_gelu_tanh {
            backend.matmul_ffn_gate_up_silu(
                &ws.residual,
                &self.w_gate,
                &self.w_up,
                &mut ws.gate,
                &mut ws.up,
            )?;
        } else {
            backend.matmul_transposed(&ws.residual, &self.w_gate, &mut ws.gate)?;
            backend.matmul_transposed(&ws.residual, &self.w_up, &mut ws.up)?;
        }
        crate::thread_pool::get_pool().end_batch();
        if is_gpu && std::env::var_os("LLMRS_DISABLE_FLUSH_FFN").is_none() {
            backend.flush()?;
        }

        // silu_mul(GELU 경로만 별도 activation) + down matmul. partition 미적용이라 항상 실행.
        if use_gelu_tanh {
            backend.gelu_tanh_mul(&mut ws.gate, &ws.up)?;
        }
        backend.matmul_transposed(&ws.gate, &self.w_down, &mut ws.down)?;

        // 10. Residual 2 — FFN 결과를 x 에 누적.
        if let Some(ref pfn) = self.post_ffn_norm {
            backend.rms_norm(&mut ws.down, pfn, rms_norm_eps, true)?;
        }
        backend.add_assign(x, &ws.down)?;

        Ok(())
    }
}

/// decode seq_len=1 의 RoPE-적용 Q `[n_heads_q*head_dim]` 를 per-kv_head `[n_kv_heads*head_dim]` 로 GQA
/// 환원한다. 식은 [`QueryStatsAccumulator::accumulate_layer`](crate::inference::query_stats) 의 per-step
/// 환원과 **동일**(kv_head 그룹 `n_rep=n_heads_q/n_kv_heads` 개 Q-head 의 element-wise 평균) — 그래서
/// faithful 현재 Q 는 QueryStats 의 1-sample(count=1) mean 과 일치하고, 누적될수록 둘이 갈린다(검증 기준).
/// 출력 레이아웃 `out[kv_head*head_dim + d]` 는 `QueryHandle::read_row(0, kv_head)` 계약과 일치.
fn gqa_reduce_query(q: &[f32], n_heads_q: usize, n_kv_heads: usize, head_dim: usize) -> Vec<f32> {
    let n_kv_heads = n_kv_heads.max(1);
    let n_rep = (n_heads_q / n_kv_heads).max(1);
    let inv_rep = 1.0 / n_rep as f32;
    let mut out = vec![0.0f32; n_kv_heads * head_dim];
    for kv_h in 0..n_kv_heads {
        let out_base = kv_h * head_dim;
        for d in 0..head_dim {
            let mut x = 0.0f32;
            for r in 0..n_rep {
                x += q[(kv_h * n_rep + r) * head_dim + d];
            }
            out[out_base + d] = x * inv_rep;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::gqa_reduce_query;
    use crate::inference::query_stats::QueryStatsAccumulator;

    /// 단순 GQA 환원 산술: n_heads_q=4, n_kv_heads=2, head_dim=2, n_rep=2.
    #[test]
    fn gqa_reduce_simple_average() {
        // Q0=[1,2] Q1=[3,4] → kv0=[2,3]; Q2=[10,20] Q3=[30,40] → kv1=[20,30].
        let q = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let out = gqa_reduce_query(&q, 4, 2, 2);
        assert_eq!(out, vec![2.0, 3.0, 20.0, 30.0]);
    }

    /// MHA (n_rep=1): 환원은 identity.
    #[test]
    fn gqa_reduce_mha_identity() {
        let q = [5.0, 6.0, 7.0, 8.0];
        let out = gqa_reduce_query(&q, 2, 2, 2);
        assert_eq!(out, vec![5.0, 6.0, 7.0, 8.0]);
    }

    /// ★정확성 기준(reference): faithful 현재 Q 의 GQA 환원은 `QueryStatsAccumulator` 가 같은 Q 를
    /// 1-sample 누적했을 때의 per-kv_head mean(row0)과 **정확히 일치**한다 — faithful Q 가 QueryStats 의
    /// count=1 mean 과 같은 환원 식을 쓰며, 차이는 오직 시간 누적(현재 vs running-mean)임을 증명한다.
    #[test]
    fn gqa_reduce_matches_query_stats_count1_mean() {
        let n_heads_q = 4;
        let n_kv_heads = 2;
        let head_dim = 2;
        let q = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];

        let faithful = gqa_reduce_query(&q, n_heads_q, n_kv_heads, head_dim);

        let mut acc = QueryStatsAccumulator::new(1, n_heads_q, n_kv_heads, head_dim);
        acc.set_active(true);
        acc.accumulate_layer(&q, 0);
        let stats = acc.layer_stats(0); // [kv_head * 2 * head_dim + row*head_dim + d], row0=mean.

        for kv_h in 0..n_kv_heads {
            for d in 0..head_dim {
                let mean = stats[kv_h * 2 * head_dim + d]; // row0 = mean.
                let faith = faithful[kv_h * head_dim + d];
                assert!(
                    (mean - faith).abs() < 1e-6,
                    "kv{kv_h} d{d}: faithful {faith} != QueryStats count=1 mean {mean}"
                );
            }
        }
    }
}
