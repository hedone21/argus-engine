//! [`ModelForward`] — first concrete [`Forward`] implementation (Phase 4-3).
//!
//! Wraps [`TransformerModel::forward_into`] for the standard `KVCache` path.
//! Owns the backend handle, model `Arc`, KV caches, decode workspace, lazy
//! prefill workspace, and two reusable logits tensors.
//!
//! Out of scope for 4-3 (kept as `None` in the forward args):
//! `skip_config`, `profiler`, `importance_collector`.
//! These are absorbed by the `PipelineStage` registry (eviction/observe stages)
//! — Phase β decode-loop rewrite (the v1 `EvictionStage`/`SwapStage`/
//! `DecodeObserver` traits were deleted in β-7).
//!
//! `layer_boundary_hook` is wired (§5.9.2 Track B): `step` reads the shared
//! `hook_cell` (installed by `WeightSwapStage::commit` in IntraForward/
//! LayerImmediate mode) and injects it into the decode forward args. When the
//! cell is `None` (production default / prefill), the slot stays `None`
//! (INV-147 zero-overhead).
//!
//! `score_accumulator` is wired (§5.9.1 Track A): `step` reads the shared
//! `score_cell` (populated by `build_bench_loop` for score-based eviction
//! policies h2o/h2o_plus/d2o), calls `begin_step()`, then injects
//! `Some(&mut acc)` into decode forward args. `end_step()` is called
//! automatically inside `forward_into` (transformer.rs:1671-1672) — caller
//! must NOT call it again. Prefill is always `None` (eval_loop.rs:240 정본).

use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::backend::Backend;
#[cfg(feature = "opencl")]
use crate::backend::opencl::plan::FullKernelPlan;
use crate::buffer::DType;
use crate::format::KVCacheFormat;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::inference::query_stats::QueryStatsAccumulator;
use crate::inference::sampling::StepCtx;
use crate::kv::kv_cache::KVCache;
use crate::kv::standard_format::StandardFormat;
use crate::kv_cache_ops::KVLayout;
use crate::layer_boundary_hook::LayerBoundaryHook;
use crate::layers::workspace::{LayerWorkspace, PrefillWorkspace, WorkspaceConfig};
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
#[cfg(feature = "opencl")]
use crate::model_config::ModelArch;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use crate::session::forward::Forward;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Standard `Forward` implementation backed by [`TransformerModel::forward_into`]
/// and a `Vec<KVCache>`.
///
/// Workspace policy (Phase 4-3 §P4 "Hybrid"):
/// - `decode_workspace` is allocated eagerly in [`Self::new`] (small,
///   `[1, 1, *]`-shaped).
/// - `prefill_workspace` is allocated lazily on the first `prefill()` call
///   (large, `[1, seq_len, *]`-shaped). Reallocated if a longer prompt
///   arrives.
pub struct ModelForward {
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    cpu_backend: Arc<dyn Backend>,
    model: Arc<TransformerModel>,
    kv_caches: Vec<KVCache>,

    // §5.9.2 Track B: assembly 가 생성해 WeightSwapStage 와 공유하는 hook cell. Stage 가 commit
    // tick 에 `Some(hook)` 설치 → `step` 이 매 decode step lock-read 후 forward args 슬롯 주입.
    // swap 미구성 조립처(chat/standard happy path)는 `Arc::new(Mutex::new(None))` 더미 — 항상 None.
    hook_cell: Arc<Mutex<Option<Arc<dyn LayerBoundaryHook>>>>,

    // §5.9.1 Track A: assembly 가 생성해 EvictionStage(scored) + CommandDispatcher 와 공유하는
    // score cell. score-based policy(h2o/h2o_plus/d2o) 구성 시에만 Some(acc) — 그 외 더미 None.
    // `step` 이 lock-read 후 acc.is_active()면 begin_step() 호출 + forward args 에 주입(decode only).
    // end_step() 은 forward_into 내부 자동 호출(transformer.rs:1671) — caller 호출 금지.
    score_cell: Arc<Mutex<Option<AttentionScoreAccumulator>>>,

    // 선택적 read stage(Quest 류). `--read-stage` 지정 시 Some, 미지정(production
    // 기본)은 None. `step`(decode)이 `as_deref()` 로 forward args 에 대여 주입한다 — None 이면
    // transformer.rs seam 의 `is_some` branch 1회 → full read 직행(INV-147 byte-identical).
    read_stage: Option<Box<dyn argus_extension_api::KVReadStage>>,

    // QueryStats(future-attention) producer. `Some` iff the installed read stage declared
    // `wants_query_stats` (set in `set_read_stage`); `step` then lends `&mut acc` to the decode
    // forward so transformer.rs accumulates the per-(layer,kv_head) running Q mean/var and feeds it
    // into the read seam (quest page scoring). `None` (production default / non-QueryStats read
    // stage) → forward arg None → seam `is_some` gate skips the per-step cost (INV-147).
    query_stats_accumulator: Option<QueryStatsAccumulator>,

    // R-P1-1 PFA producer. assembly 가 생성해 PrefillKeepSetStage 와 공유하는 cell(score_cell 패턴).
    // `wants_prefill_attn` 이면 prefill 최종 청크가 layer 별 `[n_heads_q * prefix_len]` SUM-pooled
    // attention 확률을 채워 이 cell 에 넣고, stage 가 PrefillEnd 에서 읽는다. 미무장 시 더미 None cell +
    // false → prefill 비용 0(byte-identical). `q_window` 는 plugin policy(arming 시 주입).
    prefill_attn_cell: Arc<Mutex<Option<Vec<Vec<f32>>>>>,
    wants_prefill_attn: bool,
    q_window: usize,

    decode_workspace: LayerWorkspace,
    // Phase 4-4.5: paradigm equivalence requires `prefill_workspace: None`
    // in `forward_into` args so production owned-ws path is hit. These two
    // fields are kept for future caller-reuse re-enable after the regression
    // is closed; suppress the dead-code lint until then.
    #[allow(dead_code)]
    prefill_workspace: Option<PrefillWorkspace>,
    #[allow(dead_code)]
    max_seq_len: usize,

    // Owned single-token decode input + per-token x_gen scratch + logits.
    // Allocated once to keep the vtable microbench signal clean (no per-step
    // GPU buffer creation).
    decode_input: Tensor,        // [1, 1] U8 (u32 token id)
    decode_x_gen: Tensor,        // [1, 1, hidden]
    logits_decode: Tensor,       // [1, 1, vocab]
    logits_prefill_last: Tensor, // [1, 1, vocab] (logits_last_only=true)

    vocab_size: usize,

    // fmt-cache wiring. prefill 시작 시 `kv_caches` 를 `Vec<Arc<StandardFormat>>` 로 wrap
    // (by-value move, 단일 물리 캐시) → forward/decode/eviction 모두 fmt(StandardFormat) 경로.
    // 5-F: fmt 가 production 유일 경로(OLD forward_into<C> 폐기). prefill 후 항상 Some.
    fmt_caches: Option<Vec<Arc<StandardFormat>>>,

    // Phase 4-4.7 (A1): plan-aware decode. step()이 production fallback
    // (generate.rs l.4351~4477)과 동일하게 execute_plan → forward_into fallback
    // → 다음 step lazy rebuild를 자체적으로 수행한다.
    //
    // `gpu_plan`: 현재 보유 중인 plan (lazy build, invalidation 시 None).
    // `sticky_disabled`: 한 번 build 실패 또는 invalidation lock-out 발동 시
    //   매 step rebuild를 spam하지 않도록 차단 (generate.rs l.4213 패턴).
    // `plan_enabled`: 호출자(`build_standard_loop`)가 `!args.no_gpu_plan`을
    //   전달. CLI `--no-gpu-plan` 활성 시 false → plan path 완전 우회.
    #[cfg(feature = "opencl")]
    gpu_plan: Option<FullKernelPlan>,
    #[cfg(feature = "opencl")]
    sticky_disabled: bool,
    #[cfg(feature = "opencl")]
    plan_enabled: bool,
}

impl ModelForward {
    /// Build a `ModelForward` ready to be passed to
    /// [`crate::session::DecodeLoopBuilder::with_forward`].
    ///
    /// `max_seq_len` caps the lazy `PrefillWorkspace` allocation. KV caches
    /// must already be sized for the same context window.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: Arc<dyn Backend>,
        memory: Arc<dyn Memory>,
        cpu_backend: Arc<dyn Backend>,
        model: Arc<TransformerModel>,
        kv_caches: Vec<KVCache>,
        max_seq_len: usize,
        #[cfg_attr(not(feature = "opencl"), allow(unused_variables))] plan_enabled: bool,
        // §5.9.2 Track B: WeightSwapStage 와 공유하는 layer-boundary hook cell. swap 미구성
        // 조립처는 `Arc::new(Mutex::new(None))` 더미를 넘긴다(항상 None — INV-147 거동-0).
        hook_cell: Arc<Mutex<Option<Arc<dyn LayerBoundaryHook>>>>,
        // §5.9.1 Track A: score-based eviction(h2o/h2o_plus/d2o) 공유 score cell.
        // 비-score 조립처는 `Arc::new(Mutex::new(None))` 더미를 넘긴다(항상 None).
        score_cell: Arc<Mutex<Option<AttentionScoreAccumulator>>>,
    ) -> Result<Self> {
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

        let mut s = Self {
            backend,
            memory,
            cpu_backend,
            model,
            kv_caches,
            hook_cell,
            score_cell,
            read_stage: None,
            query_stats_accumulator: None,
            prefill_attn_cell: Arc::new(Mutex::new(None)),
            wants_prefill_attn: false,
            q_window: 0,
            decode_workspace,
            prefill_workspace: None,
            max_seq_len,
            decode_input,
            decode_x_gen,
            logits_decode,
            logits_prefill_last,
            vocab_size,
            fmt_caches: None,
            #[cfg(feature = "opencl")]
            gpu_plan: None,
            #[cfg(feature = "opencl")]
            sticky_disabled: false,
            #[cfg(feature = "opencl")]
            plan_enabled,
        };
        // β-3 commit A: construction 시점 wrap — EvictionStage register 시점에
        // fmt handle 을 보유(INV-STAGE-LAYER-HANDLE). prefill/step 의 ensure_fmt_wrapped
        // 호출은 이미 Some 이라 defensive no-op 으로 비용 0.
        s.ensure_fmt_wrapped();
        Ok(s)
    }

    /// 선택적 read stage 를 주입한다(decode 시 attention 직전 read_plan 호출원).
    /// `--read-stage` 미지정이면 호출되지 않아 read_stage = None(full read, INV-147 byte-identical).
    ///
    /// `wants_query_stats`(= `KVReadStageReg::wants_query_stats`)이면 per-(layer,kv_head) Q running
    /// mean/var 를 모으는 `QueryStatsAccumulator` 를 모델 차원으로 생성·활성화해 decode 시 forward 에
    /// 대여한다(read stage 의 future-attention). false 면 미생성 → 디코드 루프 비용 0.
    pub fn set_read_stage(
        &mut self,
        stage: Box<dyn argus_extension_api::KVReadStage>,
        wants_query_stats: bool,
    ) {
        self.read_stage = Some(stage);
        if wants_query_stats {
            let cfg = &self.model.config;
            let mut acc = QueryStatsAccumulator::new(
                cfg.num_hidden_layers,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
            );
            acc.set_active(true);
            self.query_stats_accumulator = Some(acc);
        }
    }

    /// R-P1-1: prefill-end PFA producer 무장(`set_read_stage` 미러). assembly 가 PrefillKeepSetStage 와
    /// 공유하는 `cell` Arc + plugin policy `q_window` 를 주입한다. 미호출(production 기본)이면 PFA 미산출
    /// → prefill byte-identical(`wants_prefill_attn=false`).
    pub fn set_prefill_attn(&mut self, cell: Arc<Mutex<Option<Vec<Vec<f32>>>>>, q_window: usize) {
        self.prefill_attn_cell = cell;
        self.wants_prefill_attn = true;
        self.q_window = q_window;
    }

    /// Phase 4-4.7 (A1): plan eligibility 검사 + build 시도.
    ///
    /// production fallback (`generate.rs` l.4186~4199) 가드와 동치 — backend가
    /// OpenCL이고 `--no-gpu-plan` 비활성이며 Gemma3 아닐 때만. score
    /// accumulator / partition / swap_intra_forward 등 추가 가드는 호출자
    /// `is_standard_happy_path`에서 사전 차단되어 도달 시점에 모두 false 보장.
    ///
    /// 결과가 None일 때 `sticky_disabled = true`로 lock-out하여 매 step rebuild
    /// spam을 차단. invalidation 발생 시 호출자가 `gpu_plan = None`으로 set하면
    /// 다음 step 진입에서 sticky_disabled가 false인 경우에만 자동 rebuild.
    #[cfg(feature = "opencl")]
    fn try_build_plan(&mut self) -> Option<FullKernelPlan> {
        // 환경변수 `LLMRS_FWD_TRACE=1` 시 plan path 진입/거부/실패 stderr 로그.
        // Phase 4-4.7 device 측정에서 `build_plan returned None` 위치 진단용.
        // 후속 Phase 4-4.8 plan-path 진단 sprint에서 활용.
        let trace = std::env::var_os("LLMRS_FWD_TRACE").is_some();
        if !self.plan_enabled {
            if trace {
                eprintln!("[fwd-trace] skip: plan_enabled=false");
            }
            return None;
        }
        if self.sticky_disabled {
            if trace {
                eprintln!("[fwd-trace] skip: sticky_disabled");
            }
            return None;
        }
        if self.backend.name() != "OpenCL" {
            if trace {
                eprintln!("[fwd-trace] skip: backend.name()={}", self.backend.name());
            }
            return None;
        }
        if matches!(self.model.config.arch, ModelArch::Gemma3) {
            if trace {
                eprintln!("[fwd-trace] skip: arch=Gemma3");
            }
            return None;
        }
        // (3p) ④-a: `build_plan`(StandardFormat handle slice). 5-F: fmt 가 유일 경로 —
        // ensure_fmt_wrapped 가 kv_caches 를 mem::take 로 fmt_caches 로 옮겼으므로 항상 Some.
        let handles = self
            .fmt_caches
            .as_ref()
            .expect("fmt_caches Some after ensure_fmt_wrapped (5-F: fmt-only)");
        let plan = self.model.build_plan(
            &self.decode_x_gen,
            &self.logits_decode,
            &self.decode_workspace,
            handles,
            &self.backend,
        );
        if plan.is_none() {
            // build_plan이 None 반환 → 본 모델/상태에서 plan path 미지원.
            // 매 step 시도를 막기 위해 sticky lock-out.
            if trace {
                eprintln!("[fwd-trace] build_plan returned None → sticky lock");
            }
            self.sticky_disabled = true;
        } else if trace {
            eprintln!("[fwd-trace] build_plan SUCCESS");
        }
        plan
    }

    /// §5.9.2 Track B: hook cell 1회 lock-read → 설치돼 있으면 `Arc` clone 반환.
    /// 단일 스레드(INV-018)라 lock contention 0. clone 한 `Arc` 가 forward_into 동안 hook 을
    /// 살아 있게 유지하므로 guard 를 forward 호출 동안 붙들 필요가 없다(lock 즉시 해제).
    /// cell `None`(production happy/chat 더미)이면 `None` — 거동-0. 슬롯 구성 로직은
    /// `read_hook_cell` 자유 함수로 추출(ModelForward fixture 없이 host 단위테스트 가능).
    fn current_hook(&self) -> Option<Arc<dyn LayerBoundaryHook>> {
        read_hook_cell(&self.hook_cell)
    }

    /// §5.9.1 Track A: score cell 에서 active accumulator 여부 확인.
    /// active 면 `true`(plan path 우회 + begin_step + forward args 주입 필요).
    fn score_cell_active(&self) -> bool {
        read_score_cell_active(&self.score_cell)
    }

    pub fn model(&self) -> &Arc<TransformerModel> {
        &self.model
    }

    /// β-3: register 시점 Stage 가 보유할 fmt handle (INV-STAGE-LAYER-HANDLE).
    /// 빈 캐시로 구성된 경우 빈 슬라이스.
    pub fn fmt_caches(&self) -> &[Arc<StandardFormat>] {
        self.fmt_caches.as_deref().unwrap_or(&[])
    }

    /// `kv_caches` 를 `StandardFormat` 으로 1회 wrap.
    ///
    /// **construction 시점 wrap (β-3 commit A)** — `new()` 끝에서 즉시 호출.
    /// prefill/step 호출은 **defensive no-op** (fmt_caches.is_some() early return, 비용 0).
    ///
    /// **by-value move**(`mem::take`)하므로 물리 캐시는 fmt 안에 단 한 벌만 존재(dual-ownership
    /// 부재 — interior mutability 로 forward/eviction 모두 `&self` 통과). 이미 wrap /
    /// `kv_caches` 빈 경우 no-op. 5-F: fmt 가 production 유일 경로(OLD forward_into<C> 폐기).
    fn ensure_fmt_wrapped(&mut self) {
        if self.fmt_caches.is_some() || self.kv_caches.is_empty() {
            return;
        }
        let caches = std::mem::take(&mut self.kv_caches);
        self.fmt_caches = wrap_kv_caches(caches);
    }

    /// Construct the input `[1, seq_len]` U32 tensor on the active backend.
    /// CPU-side buffer is built via `Galloc` and uploaded with
    /// `backend.copy_from`, matching the existing prefill path in
    /// `generate.rs`.
    fn build_input_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        let seq_len = tokens.len();
        let cpu_buf = Galloc::new().alloc(seq_len * 4, DType::U8)?;
        // SAFETY: cpu_buf is a freshly allocated [u8] of size seq_len*4 with
        // alignment from Galloc which satisfies u32 alignment (Galloc returns
        // 64B-aligned blocks). We immediately initialise it.
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

    /// Lazy allocator for `prefill_workspace` with a seq_len realloc guard
    /// (Phase 4-3 §R4). Reuses the existing workspace when its capacity is
    /// already ≥ `seq_len`; otherwise drops and re-allocates.
    #[allow(dead_code)] // Phase 4-4.5: see struct comment.
    fn ensure_prefill_workspace(&mut self, seq_len: usize) -> Result<()> {
        let needs_alloc = match self.prefill_workspace.as_ref() {
            None => true,
            Some(ws) => ws.seq_len() < seq_len,
        };
        if needs_alloc {
            self.prefill_workspace = None; // drop old GPU buffers first
            let config = workspace_config_for(&self.model, self.max_seq_len);
            let ws = PrefillWorkspace::new(
                &config,
                seq_len.min(self.max_seq_len),
                self.memory.as_ref(),
                self.backend.clone(),
            )?;
            self.prefill_workspace = Some(ws);
        }
        Ok(())
    }

    /// Derive a safe `chunk_size` for prefill. CPU (max_single_alloc=usize::MAX)
    /// returns `seq_len` (no chunking needed). GPU mirrors the heuristic in
    /// `generate.rs::auto_gpu_chunk` — `min(budget/(vocab*4), max_alloc/(hidden*4), 512)`
    /// so neither the logits buffer nor activation buffers exceed device limits.
    fn derive_chunk_size(&self, seq_len: usize) -> usize {
        if !self.backend.is_gpu() {
            return seq_len;
        }
        let max_alloc = self.backend.max_single_alloc();
        if max_alloc == 0 || max_alloc == usize::MAX {
            return seq_len;
        }
        let hidden = self.model.config.hidden_size;
        let budget = max_alloc / 2;
        let by_vocab = (budget / (self.vocab_size * 4)).max(1);
        let by_hidden = (max_alloc / (hidden * 4)).max(1);
        by_vocab.min(by_hidden).min(512).min(seq_len)
    }

    /// Read a `[1, 1, vocab]` logits tensor off the backend into a `Vec<f32>`.
    /// Forces a backend sync first so async backends (CUDA/OpenCL) produce a
    /// stable snapshot.
    fn read_logits(&self, logits: &Tensor) -> Result<Vec<f32>> {
        self.backend.synchronize()?;
        let mut out = vec![0.0f32; self.vocab_size];
        // SAFETY: `out` is a freshly initialised f32 slice of length vocab_size;
        // reinterpreting as [u8; vocab_size*4] is sound for read_buffer (which
        // writes f32 bytes from the GPU buffer back into host memory). The
        // backend implementation does not retain the pointer past the call.
        unsafe {
            let bytes =
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, self.vocab_size * 4);
            self.backend.read_buffer(logits, bytes)?;
        }
        Ok(out)
    }
}

impl Forward for ModelForward {
    fn prefill(&mut self, tokens: &[u32], start_pos: usize) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            anyhow::bail!("ModelForward::prefill received zero tokens");
        }
        let seq_len = tokens.len();
        let chunk_size = self.derive_chunk_size(seq_len);
        // 5-F: fmt 가 유일 경로. chunk loop 전에 ensure_fmt_wrapped 로 kv_caches 를 fmt_caches 로
        // wrap(idempotent — 이후 decode step() 의 호출은 fmt_caches 이미 Some 이라 no-op).
        // 이후 각 chunk 를 forward_into(multi-token prefill batch scatter)로 처리.
        self.ensure_fmt_wrapped();

        let mut chunk_start = 0;
        while chunk_start < seq_len {
            let chunk_end = (chunk_start + chunk_size).min(seq_len);
            let chunk = &tokens[chunk_start..chunk_end];
            let input_tensor = self.build_input_tensor(chunk)?;

            // Split mutable handles to avoid double-borrowing `self` inside
            // the FnArgs literal.
            let backend = self.backend.clone();
            let memory_ref: *const dyn Memory = self.memory.as_ref();
            // SAFETY: `self.memory` is owned by `self` and lives across this
            // forward_into call; the raw pointer is dereferenced only on the
            // current stack frame.
            let memory: &dyn Memory = unsafe { &*memory_ref };

            // 5-F: fmt 가 유일 경로. ensure_fmt_wrapped 가 kv_caches 를 mem::take 로 fmt_caches 로
            // 옮겼으므로 항상 Some. concrete Arc clone → transient dyn Vec.
            let dyn_fmts: Vec<Arc<dyn KVCacheFormat>> = self
                .fmt_caches
                .as_ref()
                .expect("fmt_caches Some after ensure_fmt_wrapped (5-F: fmt-only)")
                .iter()
                .map(|f| f.clone() as Arc<dyn KVCacheFormat>)
                .collect();
            // R-P1-1: 최종 prefill 청크 + 무장 시에만 PFA buffer 할당(layer 별 `[n_heads_q * prefix_len]`,
            // pre-zero SUM 누적용). CPU(PR1)는 단일 청크라 항상 여기서 1회. forward 후 cell 적재 → stage
            // 가 PrefillEnd 에서 소비. 미무장/비최종 청크면 None → byte-identical.
            let is_final_chunk = chunk_end == seq_len;
            let q_window = self.q_window;
            let mut pfa_buf: Option<Vec<Vec<f32>>> = if self.wants_prefill_attn && is_final_chunk {
                let cfg = &self.model.config;
                let n_heads_q = cfg.num_attention_heads;
                let n_layers = cfg.num_hidden_layers;
                let prefix_len = start_pos + seq_len;
                Some(
                    (0..n_layers)
                        .map(|_| vec![0.0f32; n_heads_q * prefix_len])
                        .collect(),
                )
            } else {
                None
            };
            self.model.forward_into(TransformerModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: start_pos + chunk_start,
                fmts: &dyn_fmts,
                backend: &backend,
                memory,
                logits_out: &mut self.logits_prefill_last,
                x_gen: None,
                workspace: None,
                logits_last_only: true,
                // Phase α-K ①-c: eval feature 필드 (production 은 비활성).
                score_accumulator: None,
                query_stats_accumulator: None,
                skip_config: None,
                importance_collector: None,
                cache_self_need_scores: false,
                // §5.9.2 Track B: prefill 은 swap 금지(intra_forward_swap.rs:383 seq_len>1 가드)라
                // hook 주입 안 함 — 항상 None.
                layer_boundary_hook: None,
                read_stage: None,
                // R-P1-1: 최종 청크 무장 시에만 Some — layer loop 가 buf[i] 슬라이스에 PFA 누적.
                prefill_attn: pfa_buf.as_mut().map(|b| (b, q_window)),
            })?;
            // R-P1-1: forward 산출된 PFA buffer 를 공유 cell 에 적재(stage 가 PrefillEnd 에서 read).
            if let Some(buf) = pfa_buf {
                *self.prefill_attn_cell.lock().unwrap() = Some(buf);
            }

            chunk_start = chunk_end;
        }

        // Only the last chunk's last-token logits are kept; intermediate
        // chunks reused the same `logits_prefill_last` buffer in-place.
        self.read_logits(&self.logits_prefill_last)
    }

    fn step(&mut self, ctx: &StepCtx, token: u32) -> Result<Vec<f32>> {
        // Write the single token into the persistent decode_input buffer.
        // `write_buffer` is the same upload path used by the existing decode
        // loop in `generate.rs:2836`.
        let bytes = token.to_ne_bytes();
        self.backend.write_buffer(&mut self.decode_input, &bytes)?;

        // 5-F: fmt 가 유일 경로. plan path(execute_plan) 우선 시도 → build/invalidation 시
        // forward_into(trait object) 폴백. ensure_fmt_wrapped 가 prefill 시작에 wrap 완료.
        self.ensure_fmt_wrapped();

        // §5.9.2 Track B: hook 설치 여부 1회 read. 설치돼 있으면(IntraForward/LayerImmediate swap
        // 진행 중) plan path 를 우회한다 — plan path 는 layer loop 를 bypass 하므로 hook 의
        // wait-gate/on_layer_boundary 가 발화하지 못하고 swap 중 stale weight 를 읽을 위험이 있다.
        // forward_into(layer loop) 폴백만이 hook 을 정확히 호출한다.
        let hook = self.current_hook();

        // §5.9.1 Track A: score accumulator 활성 여부 1회 read. active 면 plan path(execute_plan)를
        // 우회한다 — plan path 는 CPU score_accumulator 슬롯을 지원하지 않는다(GPU gpu_score_acc
        // 는 plan path 지원하나 CPU-side AttentionScoreAccumulator 는 forward_into layer loop 에서만
        // 누적). accumulator 가 active 면 단일 lock 스코프에서 begin_step() 호출 + guard 해제 후
        // forward args 에 `Some(&mut acc)` 주입(end_step 은 forward_into 내부 자동 — 재호출 금지).
        let score_active = self.score_cell_active();

        // read stage 활성 시 plan path 우회. plan path(execute_plan)는 layer loop 를
        // bypass 하므로 read_plan seam(transformer.rs:1628)이 발화하지 못한다 — forward_into(layer
        // loop) 폴백만이 read_plan 을 정확히 호출한다(score_active 우회와 동형). None(production)이면
        // 비용 0(이 bool 1회 평가).
        let read_stage_active = self.read_stage.is_some();

        // (3p) ④-a plan path: fmt 핸들 기반 lazy build + execute_plan.
        // hook 설치 중 또는 score accumulator active 또는 read stage active 이면 우회.
        #[cfg(feature = "opencl")]
        if hook.is_none() && !score_active && !read_stage_active {
            if self.gpu_plan.is_none() && !self.sticky_disabled {
                self.gpu_plan = self.try_build_plan();
            }
            let plan_opt = self.gpu_plan.take();
            let plan_result = if let Some(plan) = plan_opt.as_ref() {
                let backend = self.backend.clone();
                let handles = self
                    .fmt_caches
                    .as_ref()
                    .expect("fmt_caches Some after ensure_fmt_wrapped (5-F: fmt-only)");
                self.model.execute_plan(
                    plan,
                    &self.decode_input,
                    ctx.pos,
                    &mut self.decode_x_gen,
                    handles,
                    &mut self.logits_decode,
                    &backend,
                )
            } else {
                Ok(false)
            };
            match plan_result {
                Ok(true) => {
                    self.gpu_plan = plan_opt;
                    return self.read_logits(&self.logits_decode);
                }
                Ok(false) | Err(_) => {
                    // build 실패 / invalidation — dyn 폴백으로 강하 (gpu_plan 은
                    // take() 로 이미 None, 다음 step 에서 lazy rebuild).
                }
            }
        }

        // 폴백: forward_into(trait object) — plan 미빌드(host CPU)·invalidation 경로.
        let dyn_fmts: Vec<Arc<dyn KVCacheFormat>> = self
            .fmt_caches
            .as_ref()
            .expect("fmt_caches Some after ensure_fmt_wrapped (5-F: fmt-only)")
            .iter()
            .map(|f| f.clone() as Arc<dyn KVCacheFormat>)
            .collect();
        let backend = self.backend.clone();
        let memory_ref: *const dyn Memory = self.memory.as_ref();
        // SAFETY: `self.memory` 는 self 소유, 본 call stack 동안 유효.
        let memory: &dyn Memory = unsafe { &*memory_ref };

        // §5.9.1 Track A: score accumulator begin_step + forward args 주입.
        // 단일 스레드(INV-018)라 lock contention 0. guard 를 forward_into 동안 유지해
        // `&mut acc` lifetime 을 보장한다(forward 완료 = end_step 자동 호출 후 guard 해제).
        // cell None 또는 acc 비활성이면 score_accumulator: None (거동-0).
        let mut score_guard = self.score_cell.lock().expect("score_cell mutex poisoned");
        if let Some(ref mut acc) = *score_guard
            && acc.is_active()
        {
            acc.begin_step();
        }

        // score_guard 를 유지한 채로 &mut acc 참조를 forward_into 에 주입.
        let acc_slot: Option<&mut AttentionScoreAccumulator> =
            score_guard.as_mut().filter(|acc| acc.is_active());

        // read stage 대여(없으면 None). transformer.rs:1628 seam 이 layer 당 1회
        // read_plan 을 호출한다. self.model 은 Arc(별도 필드) 라 self.read_stage 동시 immutable borrow 무충돌.
        let read_stage_slot: Option<&dyn argus_extension_api::KVReadStage> =
            self.read_stage.as_deref();

        // QueryStats producer 대여(없으면 None). 별도 필드라 위 score_guard / read_stage_slot /
        // 아래 &mut self.logits_decode 등과 disjoint borrow.
        let query_stats_slot: Option<&mut QueryStatsAccumulator> = self
            .query_stats_accumulator
            .as_mut()
            .filter(|a| a.is_active());

        self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &self.decode_input,
            start_pos: ctx.pos,
            fmts: &dyn_fmts,
            backend: &backend,
            memory,
            logits_out: &mut self.logits_decode,
            x_gen: Some(&mut self.decode_x_gen),
            workspace: Some(&mut self.decode_workspace),
            logits_last_only: false,
            // §5.9.1 Track A: active acc 면 주입, 아니면 None(거동-0). end_step() 은
            // forward_into 내부(transformer.rs:1671) 자동 — 재호출 금지.
            score_accumulator: acc_slot,
            // wants_query_stats read stage(quest) 설치 시 Some — transformer.rs seam 이
            // per-(layer,kv_head) Q running mean/var 를 누적하고 read_plan 에 공급한다(Expected
            // Attention). 미설치(production 기본/비-QueryStats read stage)는 None → seam is_some
            // 게이트가 비용 0 보장(INV-147 byte-identical).
            query_stats_accumulator: query_stats_slot,
            skip_config: None,
            importance_collector: None,
            cache_self_need_scores: false,
            // §5.9.2 Track B: hook 설치 시 layer loop 에 주입(wait-gate + on_layer_boundary).
            // `hook` Arc clone 이 본 forward_into 호출 동안 hook 을 살아 있게 유지한다.
            layer_boundary_hook: hook.as_deref(),
            // decode read-plan seam. None(production) 이면 transformer.rs seam 1회 분기.
            read_stage: read_stage_slot,
            // R-P1-1: decode 는 PFA 미산출(prefill-only producer).
            prefill_attn: None,
        })?;
        drop(score_guard); // guard 명시 해제 (end_step 이미 완료)
        self.read_logits(&self.logits_decode)
    }

    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }

    fn on_kv_prune(&mut self, _new_pos: usize) {
        // argus-bench AB-1: eviction 이 KV position 을 shift 하면 보유 중인 GPU
        // kernel plan(execute_plan 용 FullKernelPlan)이 stale offset 을 갖게 되어
        // 다음 step 에서 silent garbage 위험. plan 을 invalidate 하여 다음 step 의
        // lazy rebuild(또는 dyn 폴백)로 강하시킨다. CPU(host)는 plan 부재라 no-op.
        // fmt_caches 의 inner KVCache current_pos 는 force_evict 가 직접 갱신했고
        // (shared Arc, interior mutability) loop pos 도 new_pos 로 동기화되었으므로
        // 별도 cache 갱신은 불필요.
        #[cfg(feature = "opencl")]
        {
            self.gpu_plan = None;
        }
    }

    fn reset_kv(&mut self) -> anyhow::Result<()> {
        // fmt 활성 시 inner cache 는 StandardFormat 안 → with_cache_mut seam 으로 reset.
        if let Some(fmts) = &self.fmt_caches {
            for f in fmts {
                f.with_cache_mut(|c| c.current_pos = 0);
            }
        } else {
            for cache in &mut self.kv_caches {
                cache.current_pos = 0;
            }
        }
        Ok(())
    }

    fn try_evict(
        &mut self,
        cache_manager: &crate::kv::cache_manager::CacheManager,
        scores: Option<&[f32]>,
        last_attn: Option<&[f32]>,
        force: bool,
        target_ratio: f32,
    ) -> anyhow::Result<(usize, usize)> {
        // Phase α-K BC (3d): fmt 활성(chat fmt-wrap) 시 UER(Unwrap-Evict-Rewrap).
        // fmt-wrap 이 kv_caches 를 mem::take 해 비웠으므로 OLD 경로는 빈 슬라이스 → silent no-op.
        // inner KVCache 들을 연속 Vec 로 꺼내(take_inner) OLD `cache_manager.{force,maybe}_evict*`
        // 를 **그대로 재사용**(전 정책 sliding/h2o/h2o_plus/d2o + weighted-merge cross-layer merge + execute_dispatch
        // 의 madvise/new_pos/CacheEvent 보존, selection 동일성 = code-path 동일성) 후 다시 넣는다(put_inner).
        // 설계: design_alpha_k_3d_chat_fmt_2026_06_04.md (Approach B, 적대검증 3 lens 만장일치).
        if let Some(fmts) = &self.fmt_caches {
            // W1 불변식: fmts = ensure_fmt_wrapped enumerate 순서 == layer idx (weighted-merge cross-layer 전제).
            let before_pos = fmts
                .first()
                .map(|f| f.with_cache_mut(|c| c.current_pos))
                .unwrap_or(0);
            let mut temp: Vec<crate::kv::kv_cache::KVCache> =
                fmts.iter().map(|f| f.take_inner()).collect();
            // evict 결과를 캡처 → `?` 전파를 rewrap 이후로 미뤄 placeholder 잔존 방지(잔여위험 1).
            let evict_result = if force {
                match scores {
                    Some(sc) => cache_manager.force_evict_with_scores(
                        &mut temp,
                        target_ratio,
                        sc,
                        last_attn,
                    ),
                    None => cache_manager.force_evict(&mut temp, target_ratio),
                }
            } else {
                match scores {
                    Some(sc) => cache_manager.maybe_evict_with_scores(&mut temp, sc, last_attn),
                    None => cache_manager.maybe_evict(&mut temp),
                }
            };
            // rewrap: 항상 실행(Err/Ok 무관) — inner 복귀, placeholder 폐기.
            for (f, c) in fmts.iter().zip(temp) {
                f.put_inner(c);
            }
            let result = evict_result?;
            return if result.evicted {
                Ok((before_pos.saturating_sub(result.new_pos), result.new_pos))
            } else {
                Ok((0, before_pos))
            };
        }

        let before_pos = self.kv_caches.first().map(|c| c.current_pos).unwrap_or(0);

        let result = if force {
            match scores {
                Some(sc) => cache_manager.force_evict_with_scores(
                    &mut self.kv_caches,
                    target_ratio,
                    sc,
                    last_attn,
                )?,
                None => cache_manager.force_evict(&mut self.kv_caches, target_ratio)?,
            }
        } else {
            match scores {
                Some(sc) => {
                    cache_manager.maybe_evict_with_scores(&mut self.kv_caches, sc, last_attn)?
                }
                None => cache_manager.maybe_evict(&mut self.kv_caches)?,
            }
        };

        if result.evicted {
            let removed = before_pos.saturating_sub(result.new_pos);
            Ok((removed, result.new_pos))
        } else {
            Ok((0, before_pos))
        }
    }

    /// prefix cache save (ENG-085, INV-189). prefill 완료 직후 호출됨.
    ///
    /// fmt_caches(StandardFormat)가 있으면 `SnapshotRestore::snapshot_prefix` 경유 직렬화.
    /// opaque/quant-window(fmt_caches == None 또는 opaque buffer) 는 no-op (INV-190 정책 동치 — 에러 아님).
    fn save_kv_prefix(
        &self,
        path: &std::path::Path,
        model_hash: &[u8; 32],
        tokenizer_hash: &[u8; 32],
        token_ids: &[u32],
        last_logits: &[f32],
        backend: &dyn crate::backend::Backend,
    ) -> anyhow::Result<()> {
        use crate::format::SnapshotRestore;
        use crate::session::prefix_cache::save_prefix;

        let Some(fmts) = &self.fmt_caches else {
            // kv_caches 미wrap (prefill 전 호출 혹은 opaque) → no-op
            return Ok(());
        };
        if fmts.is_empty() {
            return Ok(());
        }
        // 첫 번째 cache의 geometry를 사용
        let (kv_heads, head_dim) = fmts
            .first()
            .map(|f| f.with_cache_mut(|c| (c.kv_heads() as u32, c.head_dim() as u32)))
            .unwrap_or((0, 0));
        let format_id = fmts.first().map(|f| f.snapshot_format_id()).unwrap_or(0);

        let snap_refs: Vec<&dyn SnapshotRestore> = fmts
            .iter()
            .map(|f| f.as_ref() as &dyn SnapshotRestore)
            .collect();

        save_prefix(
            path,
            model_hash,
            tokenizer_hash,
            token_ids,
            last_logits,
            format_id,
            &snap_refs,
            kv_heads,
            head_dim,
            backend,
        )
    }
}

/// §5.9.2 Track B: hook cell → forward args 슬롯값(Arc clone). 설치돼 있으면 `Some(Arc)`,
/// 미설치(`None`)면 `None`. `current_hook` 의 슬롯 구성 로직 — ModelForward fixture 없이 host
/// 단위테스트가 가능하도록 자유 함수로 추출(단일 스레드 INV-018 → lock contention 0).
fn read_hook_cell(
    cell: &Arc<Mutex<Option<Arc<dyn LayerBoundaryHook>>>>,
) -> Option<Arc<dyn LayerBoundaryHook>> {
    cell.lock()
        .expect("model_forward hook_cell mutex poisoned")
        .clone()
}

/// §5.9.1 Track A: score cell 의 active 여부 — `Some(acc)` 이고 `acc.is_active()` 면 `true`.
/// plan path 우회 판단 + step 로직에서 사용. ModelForward fixture 없이 host 단위테스트 가능하도록
/// 자유 함수로 추출(read_hook_cell 패턴 동형).
fn read_score_cell_active(cell: &Arc<Mutex<Option<AttentionScoreAccumulator>>>) -> bool {
    cell.lock()
        .expect("score_cell mutex poisoned")
        .as_ref()
        .is_some_and(|acc| acc.is_active())
}

/// `Vec<KVCache>` → `Vec<Arc<StandardFormat>>` wrap (by-value move, 단일 물리 캐시).
///
/// 빈 입력이면 `None` (기존 `kv_caches.is_empty()` 가드 등가).
/// W1 불변식: enumerate 순서 == layer idx (weighted-merge cross-layer 전제).
pub(crate) fn wrap_kv_caches(caches: Vec<KVCache>) -> Option<Vec<Arc<StandardFormat>>> {
    if caches.is_empty() {
        return None;
    }
    let fmts: Vec<Arc<StandardFormat>> = caches
        .into_iter()
        .enumerate()
        .map(|(i, c)| Arc::new(StandardFormat::new(i, c)))
        .collect();
    if std::env::var_os("LLMRS_FWD_TRACE").is_some() {
        eprintln!(
            "[fwd-trace] fmt default: wrapped {} KVCache → StandardFormat (decode = forward_into)",
            fmts.len()
        );
    }
    Some(fmts)
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

/// Allocate a standard `KVCache` per layer using the same recipe as
/// `generate.rs:406` — `HeadMajor` layout, dynamic grow, `kv_buf_size`
/// derived from `dtype`. Exposed for `bin/probe_inference_loop.rs` so the
/// microbench does not need to copy this block.
pub fn alloc_standard_kv_caches(
    model: &TransformerModel,
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    initial_capacity: usize,
    max_seq_len: usize,
    dtype: DType,
) -> Result<Vec<KVCache>> {
    let num_layers = model.config.num_hidden_layers;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;

    let n_values = initial_capacity * kv_heads * head_dim;
    let kv_buf_size = match dtype {
        DType::Q4_0 => {
            use crate::quant::{BlockQ4_0, QK4_0};
            (n_values / QK4_0) * std::mem::size_of::<BlockQ4_0>()
        }
        _ => n_values * dtype.size(),
    };

    let mut caches = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let k_buf = memory.alloc_kv(kv_buf_size, dtype)?;
        let v_buf = memory.alloc_kv(kv_buf_size, dtype)?;
        let shape = Shape::new(vec![1, kv_heads, initial_capacity, head_dim]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend.clone());
        caches.push(
            KVCache::new_dynamic(
                k,
                v,
                initial_capacity,
                max_seq_len,
                kv_heads,
                head_dim,
                memory.clone(),
            )
            .with_layout(KVLayout::HeadMajor),
        );
    }
    Ok(caches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    /// F32 SeqMajor KVCache 를 테스트용으로 구성 (standard_format.rs 테스트 패턴 차용).
    fn make_cache_with_pos(kv_heads: usize, head_dim: usize, pos: usize) -> KVCache {
        let max_seq = 64usize;
        let total = max_seq * kv_heads * head_dim;
        let buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let k = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            buf.clone(),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, max_seq, kv_heads, head_dim]),
            buf,
            backend,
        );
        let mut c = KVCache::new(k, v, max_seq);
        c.current_pos = pos;
        c
    }

    /// 빈 입력 → None (기존 is_empty() 가드 보존).
    #[test]
    fn wrap_empty_returns_none() {
        let result = wrap_kv_caches(vec![]);
        assert!(result.is_none());
    }

    /// KVCache 3개 wrap → handles[i].with_cache_mut current_pos == i+1, 순서 보존.
    #[test]
    fn wrap_preserves_layer_order_and_pos() {
        let caches = vec![
            make_cache_with_pos(2, 8, 1),
            make_cache_with_pos(2, 8, 2),
            make_cache_with_pos(2, 8, 3),
        ];
        let handles = wrap_kv_caches(caches).expect("non-empty should return Some");
        assert_eq!(handles.len(), 3);
        for (i, h) in handles.iter().enumerate() {
            let pos = h.with_cache_mut(|c| c.current_pos);
            assert_eq!(pos, i + 1, "layer {} pos mismatch", i);
        }
    }

    /// wrap 후 handle 경유 reset → current_pos == 0 (chat reset_kv fmt-경로 단위 등가).
    #[test]
    fn wrap_handle_reset_roundtrip() {
        let caches = vec![make_cache_with_pos(2, 8, 42)];
        let handles = wrap_kv_caches(caches).expect("non-empty should return Some");
        handles[0].with_cache_mut(|c| c.current_pos = 0);
        let pos = handles[0].with_cache_mut(|c| c.current_pos);
        assert_eq!(pos, 0);
    }

    /// §5.9.2 Track B: cell `None`(미설치/더미)이면 슬롯값 `None` (production happy/chat 거동-0).
    #[test]
    fn read_hook_cell_none_yields_none() {
        let cell: Arc<Mutex<Option<Arc<dyn LayerBoundaryHook>>>> = Arc::new(Mutex::new(None));
        let slot = read_hook_cell(&cell);
        assert!(slot.is_none(), "미설치 cell → 슬롯 None");
        // `.as_deref()` 도 None (forward args 주입값).
        assert!(slot.as_deref().is_none());
    }

    /// §5.9.2 Track B: cell `Some(hook)` 이면 슬롯값 `Some` → forward args 슬롯 주입 대상.
    #[test]
    fn read_hook_cell_some_yields_some() {
        use crate::layer_boundary_hook::NoOpHook;
        let cell: Arc<Mutex<Option<Arc<dyn LayerBoundaryHook>>>> = Arc::new(Mutex::new(Some(
            Arc::new(NoOpHook) as Arc<dyn LayerBoundaryHook>,
        )));
        let slot = read_hook_cell(&cell);
        assert!(slot.is_some(), "설치 cell → 슬롯 Some");
        // `.as_deref()` 가 forward args 의 `layer_boundary_hook: Some(&dyn ...)` 를 만든다.
        assert!(slot.as_deref().is_some());
    }

    /// §5.9.1 Track A: score_cell None → read_score_cell_active = false (더미 셀, happy/chat 거동-0).
    #[test]
    fn score_cell_none_is_inactive() {
        let cell: Arc<Mutex<Option<AttentionScoreAccumulator>>> = Arc::new(Mutex::new(None));
        assert!(!read_score_cell_active(&cell), "None 셀 → inactive");
    }

    /// §5.9.1 Track A: score_cell Some(acc, active=false) → read_score_cell_active = false.
    #[test]
    fn score_cell_some_inactive_is_false() {
        let acc = AttentionScoreAccumulator::new(64, 8, 1, 0, 0.0);
        // active 기본값은 false
        let cell = Arc::new(Mutex::new(Some(acc)));
        assert!(!read_score_cell_active(&cell), "active=false 셀 → inactive");
    }

    /// §5.9.1 Track A: score_cell Some(acc, active=true) → read_score_cell_active = true.
    #[test]
    fn score_cell_some_active_is_true() {
        let mut acc = AttentionScoreAccumulator::new(64, 8, 1, 0, 0.0);
        acc.set_active(true);
        let cell = Arc::new(Mutex::new(Some(acc)));
        assert!(read_score_cell_active(&cell), "active=true 셀 → active");
    }
}
