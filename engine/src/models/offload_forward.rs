//! Offload-path forward (`forward_into_offload`) — KV cache offload 전용 forward 경로.
//!
//! `TransformerModel` 의 inherent impl 을 분할(Rust split inherent impl)해 offload(L3
//! pressure/kv) concrete 결합(`OffloadKVCache`/`OffloadFormat`/`PrefetchController`)을
//! inference 본체(`transformer.rs`)에서 격리한다 (§13.8-O 갈래 2 offload 분리 backlog).
//! 본 모듈은 offload concrete 타입을 직접 참조하므로 cross-L3 vocabulary marker 가
//! 잔존한다 — happy-path forward(`forward_into`)는 본 모듈을 경유하지 않아 무관하다.
//!
//! 분리는 **순수 모듈 이동** — `forward_into_offload` 본문은 1바이트도 변경되지 않았다
//! (byte-identical). `gather_embed`/`validate_read_plan`/`lm_head_matmul_cpu` 는
//! `transformer.rs` 에 잔존(happy-path 와 공유)하며 `pub(crate)` 로 호출한다.

use std::sync::Arc;

use anyhow::Result;

use crate::buffer::DType;
use crate::layers::transformer_layer::TransformerLayer;
use crate::model_config::ModelArch;
use crate::models::transformer::{OffloadForwardArgs, TransformerModel, is_local_layer};
use crate::shape::Shape;
use crate::tensor::Tensor;
// PreloadAccess/PreloadResult 는 L2(`crate::preload_access`) — cross-L3 아님(§13.8-O 격상).
use crate::preload_access::{PreloadAccess, PreloadResult};

impl TransformerModel {
    /// offload KV forward 경로 — `OffloadKVCache` 를 `KVCacheFormat` trait object 로 구동한다.
    ///
    /// `OffloadForward`(`session/forward/offload_forward.rs`)가 매 prefill/decode 마다 `wrap_caches`
    /// 로 만든 `Vec<Arc<OffloadFormat>>` 를 넘겨 호출하는 offload 의 **유일 live 경로**다(별도 env
    /// 게이트나 OLD generic fork 없음). 루프 골격(embedding / preload pool / depth loop / retain /
    /// release / final norm+head)은 standard forward 와 동형이고, KV 접근 두 가지만 다르다:
    ///   1. KV 를 별도 `fmts: &[Arc<OffloadFormat>]` 파라미터로 받는다 — preload/retain/release 는
    ///      `OffloadFormat` 의 interior-mut(`&self`) 메서드 경유.
    ///   2. forward 위임: decode → `forward_gen_fmt`(`&Arc<dyn KVCacheFormat>` + LayerWorkspace),
    ///      prefill → owned `PrefillWorkspace` + `forward_prefill_fmt`(forward_into:2096 미러).
    ///
    /// `dyn_fmts` = `fmts` 를 `Arc<dyn KVCacheFormat>` 로 업캐스트한 Vec(루프 전 1회) — fmt fork 가
    /// `&Arc<dyn KVCacheFormat>` 를 요구. `OffloadFormat` 은 interior-mut 라 preload pool 의 raw cast 가
    /// `*const` 이며 Mutex 가 aliasing 을 흡수한다. need_scores=false 고정(offload score 미사용,
    /// OLD :3875 일치). 마지막 pending drain 으로 모든 background task 가 종료 전 완료/drop 됨을 보장
    /// (caller 의 `Arc::try_unwrap` 성공 전제).
    // LAYER-EXEMPT: cross_l3_vocabulary — §13.8-O offload-path concrete OffloadFormat + PrefetchController (offload 분리 backlog)
    pub fn forward_into_offload(
        &self,
        args: OffloadForwardArgs<'_>,
        fmts: &[Arc<crate::kv::offload_format::OffloadFormat>],
        prefetch: &mut crate::kv::offload::prefetch::PrefetchController,
    ) -> Result<()> {
        use crate::kv::offload_format::{OffloadFormat, preload_offload_fmt_erased};
        use crate::layers::workspace::{PrefillWorkspace, WorkspaceConfig as WsCfg};

        let input_tokens = args.input_tokens;
        let start_pos = args.start_pos;
        let backend = args.backend;
        let memory = args.memory;
        let logits_out = args.logits_out;
        let x_gen = args.x_gen;
        let mut workspace = args.workspace;
        // S6: read_stage 슬롯 — decode arm 에서 layer i 완료 후 read_plan 산출 + prefetch hint 공급.
        // read_stage=None(기본)이면 아래 is_some 분기 1회만 추가(INV-147 byte-identical).
        let read_stage = args.read_stage;

        if let Some(ws) = workspace.as_deref_mut() {
            ws.reset_partition_prev();
        }

        let batch_size = input_tokens.shape().dims()[0];
        let seq_len = input_tokens.shape().dims()[1];
        let hidden_size = self.config.hidden_size;
        let is_decode = seq_len == 1;

        // 1. Embedding lookup (forward_into_offload 와 동일).
        let mut x = if is_decode {
            if let Some(xb) = x_gen {
                (*xb).clone()
            } else {
                let x_buf = memory.alloc(batch_size * seq_len * hidden_size * 4, DType::F32)?;
                Tensor::new(
                    Shape::new(vec![batch_size, seq_len, hidden_size]),
                    x_buf,
                    backend.clone(),
                )
            }
        } else {
            let x_buf = memory.alloc(batch_size * seq_len * hidden_size * 4, DType::F32)?;
            Tensor::new(
                Shape::new(vec![batch_size, seq_len, hidden_size]),
                x_buf,
                backend.clone(),
            )
        };
        self.gather_embed(input_tokens, &mut x, backend)?;
        if let Some(scale) = self.config.embed_scale {
            backend.scale(&mut x, scale)?;
        }

        let is_gemma3 = self.config.arch == ModelArch::Gemma3;
        let num_layers = self.layers.len();
        let depth = prefetch.depth();

        // fmt fork 는 `&Arc<dyn KVCacheFormat>` 를 요구 — 루프 전 1회 업캐스트.
        let dyn_fmts: Vec<Arc<dyn crate::format::KVCacheFormat>> = fmts
            .iter()
            .map(|a| a.clone() as Arc<dyn crate::format::KVCacheFormat>)
            .collect();

        // prefill owned PrefillWorkspace (forward_into:2096 미러). decode 는 args.workspace 사용,
        // prefill 은 owned alloc(forward_prefill_fmt 가 항상 PrefillWorkspace 요구).
        let ws_cfg = WsCfg {
            batch_size,
            dim: hidden_size,
            q_dim: self.config.num_attention_heads * self.config.head_dim,
            k_dim: self.config.num_key_value_heads * self.config.head_dim,
            v_dim: self.config.num_key_value_heads * self.config.head_dim,
            ffn_hidden: self.config.intermediate_size,
            n_heads: self.config.num_attention_heads,
            max_seq_len: 0,
        };
        let mut owned_prefill_ws: Option<PrefillWorkspace> = None;
        let mut needs_ws_sync = false;
        if !is_decode {
            owned_prefill_ws = Some(PrefillWorkspace::new(
                &ws_cfg,
                seq_len,
                memory,
                backend.clone(),
            )?);
            needs_ws_sync = backend.is_gpu();
        }

        // Per-token weight snapshot (forward_into_offload :3777 미러).
        let layer_snapshots: Vec<Arc<TransformerLayer>> =
            self.layers.iter().map(|s| s.load_weights()).collect();

        // Lazy-init persistent thread pool (forward_into_offload :3783 미러).
        let pool = self.preload_pool.get_or_init(|| {
            Box::new(crate::kv::offload::preload_pool::PreloadPool::new(
                prefetch.max_depth(),
            )) as Box<dyn PreloadAccess>
        });

        // 2. Synchronous initial preload: layers [0..depth).
        for fmt in fmts.iter().take(depth.min(num_layers)) {
            fmt.preload_locked()?;
        }

        // pending[j] = receiver for fmts[j]'s preload (forward_into_offload :3804 미러).
        // SAFETY: OffloadFormat 은 interior-mut(Mutex)라 raw cast 가 *const 여도 aliasing 안전.
        // far_idx != i (far_idx = i + depth, depth >= 1) — retain/release 로직 보존용 불변식.
        let mut pending: Vec<Option<std::sync::mpsc::Receiver<PreloadResult>>> =
            (0..num_layers).map(|_| None).collect();

        // Fire initial background preloads for layers [depth..2*depth).
        #[allow(clippy::needless_range_loop)]
        for j in depth..(2 * depth).min(num_layers) {
            pending[j] = Some(unsafe {
                pool.submit_raw(
                    Arc::as_ptr(&fmts[j]) as *mut OffloadFormat as *mut (),
                    preload_offload_fmt_erased,
                )
            });
        }

        // B-1 (적대검증): background preload worker 는 `Arc::as_ptr` 로 얻은 **raw pointer**(strong
        // count 미증가)로 OffloadFormat 을 deref/lock 한다. 레이어 forward 의 `?` early-return(또는
        // 패닉)이 아래 pending drain 을 건너뛰면, caller(`unwrap_caches`)가 `result?` 이전에
        // `Arc::try_unwrap`(strong=1 → 성공)+drop 으로 OffloadFormat·Mutex 를 free → in-flight worker 가
        // freed/locked 메모리를 만져 UAF / drop-while-locked UB. `DrainGuard` 가 모든 반환 경로(에러·
        // 패닉)에서 `recv()` 로 worker 완료를 보장해 happy-path 와 동일한 건전성을 회복한다.
        struct DrainGuard<'a> {
            pending: &'a mut Vec<Option<std::sync::mpsc::Receiver<PreloadResult>>>,
        }
        impl Drop for DrainGuard<'_> {
            fn drop(&mut self) {
                for slot in self.pending.iter_mut() {
                    if let Some(rx) = slot.take() {
                        let _ = rx.recv();
                    }
                }
            }
        }
        let guard = DrainGuard {
            pending: &mut pending,
        };

        // 3. Layer loop.
        for i in 0..num_layers {
            // Collect preload result for layer i.
            if let Some(rx) = guard.pending[i].take() {
                match rx.recv() {
                    Ok(PreloadResult {
                        result: Ok(()),
                        duration,
                    }) => {
                        prefetch.record_preload(duration);
                    }
                    Ok(PreloadResult { result: Err(e), .. }) => {
                        log::warn!("L{i} preload failed: {e}, falling back to sync");
                    }
                    Err(_) => {
                        log::error!("L{i} preload worker dropped result channel");
                    }
                }
            }

            // Fire preload for layer i + depth.
            let far_idx = i + depth;
            if far_idx < num_layers && guard.pending[far_idx].is_none() {
                guard.pending[far_idx] = Some(unsafe {
                    pool.submit_raw(
                        Arc::as_ptr(&fmts[far_idx]) as *mut OffloadFormat as *mut (),
                        preload_offload_fmt_erased,
                    )
                });
            }

            // Forward current layer.
            let fwd_t0 = std::time::Instant::now();
            let rope_theta_i = if is_gemma3 && is_local_layer(i, self.config.sliding_window_pattern)
            {
                self.config
                    .rope_local_theta
                    .unwrap_or(self.config.rope_theta) as f32
            } else {
                self.config.rope_theta as f32
            };
            let is_local_i = if is_gemma3 {
                Some(is_local_layer(i, self.config.sliding_window_pattern))
            } else {
                None
            };
            let layer_arc = layer_snapshots[i].clone();

            if is_decode && workspace.is_some() {
                let ws = workspace
                    .as_deref_mut()
                    .expect("decode arm: workspace.is_some() checked just before");
                // S6: read_stage=Some 이면 layer i 의 fmt 가 SelectiveRead 를 지원할 때 read_plan 을
                // 산출해 Precomputed 라우팅으로 forward_gen_fmt 에 전달 + plan.select 를 layer i+1 prefetch
                // 우선 힌트로 저장한다(D3). offload 는 plan 을 forward_gen_fmt *이전*에 돌려야(prefetch
                // 선행) 해서 현재 Q 가 아직 없다 → query=None(query-agnostic proxy). standard decode 의
                // faithful current-Q 경로(ReadRouting::Faithful)와 구분된다.
                // ⚠ 현재 production: 이 bin 의 OffloadForward 세션은 read_stage=None 으로 배선(offload_forward.rs
                // session) + OffloadFormat 은 as_selective_read()==None → 이 Precomputed 경로는 아직
                // live consumer 가 없다(read_stage 배선 + OffloadFormat SelectiveRead 구현 시 활성).
                let read_plan = read_stage.and_then(|rs| {
                    dyn_fmts[i]
                        .as_selective_read()
                        .and_then(|sr| sr.read_plan(rs, i, None, None))
                });
                let read_routing = read_plan.as_ref().and_then(|plan| {
                    Self::validate_read_plan(plan, dyn_fmts[i].current_pos()).map(|sel| {
                        crate::layers::transformer_layer::ReadRouting::Precomputed {
                            select: sel,
                            granularity: plan.granularity,
                        }
                    })
                });
                // S6 prefetch 보강 채널: plan 이 있으면 next layer 의 prefetch 우선 힌트를 저장.
                // 현재 preload_locked 는 선입선출이라 hint 가 I/O 순서를 변경하지 않는다.
                // 향후 preload_locked 에 page-id 우선 지원 추가 시 take_priority_hint 를 확장한다.
                if let (Some(_rs), Some(plan)) = (read_stage, read_plan.as_ref()) {
                    let next = i + 1;
                    if next < num_layers {
                        prefetch.set_priority_hint(next, plan.select.clone());
                    }
                }
                layer_arc.forward_gen_fmt(crate::layers::transformer_layer::ForwardGenFmtArgs {
                    x: &mut x,
                    fmt: &dyn_fmts[i],
                    start_pos,
                    backend,
                    ws,
                    rms_norm_eps: self.config.rms_norm_eps as f32,
                    rope_theta: rope_theta_i,
                    need_scores: false,
                    head_dim: self.config.head_dim,
                    skip_attn: false,
                    skip_mlp: false,
                    rms_norm_add_unit: is_gemma3,
                    use_gelu_tanh: is_gemma3,
                    is_local_attn: is_local_i,
                    local_attn_window: self.config.sliding_window,
                    layer_idx: i,
                    read_routing,
                })?;
            } else {
                // prefill(seq_len>1) 또는 **발산 A**(seq_len==1 + workspace=None). 후자는 BOS-only
                // 첫 prefill(chat repl.rs:89 → OffloadForward::prefill(tokens=[bos], workspace=None))
                // 경로 — OLD forward_into_offload 는 layer.forward 가 seq_len==1+workspace=None 에서
                // forward_prefill 로 fall-through(transformer_layer.rs:261, degenerate 1-token).
                // 둘 다 forward_prefill_fmt(owned PrefillWorkspace) 로 미러(forward_into:2141 동형).
                // owned_prefill_ws 는 seq_len>1 이면 루프 전 alloc(:4025), 발산 A 면 여기서 lazy alloc.
                if owned_prefill_ws.is_none() {
                    owned_prefill_ws = Some(PrefillWorkspace::new(
                        &ws_cfg,
                        seq_len,
                        memory,
                        backend.clone(),
                    )?);
                    needs_ws_sync = backend.is_gpu();
                }
                let pws = owned_prefill_ws
                    .as_mut()
                    .expect("prefill/divergence-A PrefillWorkspace just allocated");
                layer_arc.forward_prefill_fmt(
                    crate::layers::transformer_layer::ForwardPrefillFmtArgs {
                        x: &mut x,
                        fmt: &dyn_fmts[i],
                        start_pos,
                        backend,
                        pws,
                        rms_norm_eps: self.config.rms_norm_eps as f32,
                        rope_theta: rope_theta_i,
                        head_dim: self.config.head_dim,
                        batch_size,
                        seq_len,
                        dim: hidden_size,
                        skip_attn: false,
                        skip_mlp: false,
                        rms_norm_add_unit: is_gemma3,
                        use_gelu_tanh: is_gemma3,
                        is_local_attn: is_local_i,
                        local_attn_window: self.config.sliding_window,
                        // R-P1-1: offload forward 경로는 PFA 미산출(StandardFormat producer 전용).
                        pfa_target: None,
                        pfa_per_row_target: None,
                    },
                )?;
            }
            let fwd_dur = fwd_t0.elapsed();
            prefetch.record_forward(fwd_dur);

            // Cross-token retention (forward_into_offload :3892 미러).
            if i < depth {
                fmts[i].retain_locked();
            }

            // Release consumed layer's buffers (forward_into_offload :3897 미러).
            if i > 0 && (i - 1) >= depth {
                fmts[i - 1].release_locked();
            }
        }

        // Collect any remaining pending preloads (forward_into_offload :3904 미러).
        // ★ Arc::try_unwrap(caller) 성공·건전성 위해 모든 background worker 가 함수 반환 전 완료돼야
        // 한다 — 정상 종료는 여기서 명시 drop, 에러/패닉 early-return 은 `DrainGuard::drop` 이 동일
        // 보장(B-1). drop 후 `pending` 는 비어 있다(전부 take 됨).
        drop(guard);

        // Release last layer's buffers (forward_into_offload :3909 미러).
        if num_layers >= 1 && (num_layers - 1) >= depth {
            fmts[num_layers - 1].release_locked();
        }

        prefetch.adjust();

        // 4. Final Norm + Head.
        backend.rms_norm(
            &mut x,
            &self.norm,
            self.config.rms_norm_eps as f32,
            is_gemma3,
        )?;
        // ★W-1(적대검증): prefill(seq_len>1)+logits_last_only 면 마지막 토큰 hidden 만 head 에 통과
        // (forward_into:2311 미러). OLD forward_into_offload 의 tail 은 이 분기가 없어 logits_out=
        // [1,1,vocab](OffloadForward::prefill 할당)에 [1,seq_len,vocab] 를 써 heap overflow 했다 —
        // 신규 함수가 OOB write 하지 않도록 정정(decode seq_len=1 은 분기 미진입 → byte-불변).
        if args.logits_last_only && seq_len > 1 {
            let last_offset = (seq_len - 1) * hidden_size;
            let last_buf = memory.alloc(hidden_size * 4, DType::F32)?;
            let mut x_last = Tensor::new(
                Shape::new(vec![1, 1, hidden_size]),
                last_buf,
                backend.clone(),
            );
            backend.copy_slice(&x, &mut x_last, last_offset, 0, hidden_size)?;
            if self.lm_head_on_cpu {
                self.lm_head_matmul_cpu(&x_last, logits_out, backend)?;
            } else {
                backend.matmul_transposed(&x_last, &self.lm_head, logits_out)?;
            }
        } else if self.lm_head_on_cpu {
            self.lm_head_matmul_cpu(&x, logits_out, backend)?;
        } else {
            backend.matmul_transposed(&x, &self.lm_head, logits_out)?;
        }

        // prefill owned PrefillWorkspace drop 전 GPU 커널 완료 보장 (forward_into:2333 미러).
        if needs_ws_sync {
            backend.synchronize()?;
        }

        Ok(())
    }
}
