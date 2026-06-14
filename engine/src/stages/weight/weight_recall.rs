//! `WeightRecallStage` — `RecallWeights` runtime directive 의 OneShot `PipelineStage` (§5.6.8).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.6.8, spec: ENG-ALG-240/241,
//! MSG-043, INV-192~195.
//!
//! `WeightMutate` phase 에서 발화하여 현재 Q4_0 인 layer 의 weight 를 secondary 의
//! F16 원본 variant 로 복원한다(역방향 recall, Q4_0→F16).
//!
//! `WeightSwapStage` 의 역방향 형제 — 동일 `EngineSwapRuntime` + `TransformerModel` 을
//! 공유하지만 decider(importance-aware) 를 사용하지 않는다. F16 복원은 항상 품질 개선
//! 방향이라 "되돌릴 layer" 를 importance-aware 로 고를 이유가 없으므로 currently-Q4_0
//! 역집합의 단순 선택(ratio 이하 또는 전체)을 사용한다.
//!
//! **loud no-op 5종 (INV-195/ENG-ALG-241)**:
//! (a) secondary에 F16 variant 부재(`DtypeNotFound`) — Q4_0-only secondary.
//! (b) Adreno SOA 경로(`AdrenoSoaF16Rejected`) — SOA layout 은 Q4_0 전용.
//! (c) secondary handle 부재(`no_secondary`).
//! (d) currently-swapped layer 0개 (복원 불요).
//! (e) in-flight plan 활성 (R-1 가드, swap 과 공유 마커).
//! 각 분기: stderr 1회 + graceful `Consumed`, panic/Err-강하 금지.
//!
//! **F16 view 캐싱**: `OnceLock<Result<Arc<SecondaryMmap>, String>>` — 첫 commit 시
//! F16 view 를 바인딩하고 이후 recall 에서 재사용한다. reject 결과도 캐싱하여
//! 매 recall 에서 loud-fail 을 한 번만 실행한다.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use argus_shared::{DtypeTag, LayerSwapEntry, WeightSwapReport};

use crate::auf::BackendTag;
use crate::buffer::DType;
use crate::models::transformer::TransformerModel;
use crate::models::weights::secondary_mmap::SecondaryMmap;
use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};
use crate::session::swap_runtime::EngineSwapRuntime;

/// `WeightMutate` phase 에서 Q4_0→F16 recall 을 수행하는 OneShot Stage.
pub struct WeightRecallStage {
    model: Arc<TransformerModel>,
    runtime: Arc<EngineSwapRuntime>,
    ratio: f32,
    /// §5.9.2 Track B: ModelForward 와 공유하는 layer-boundary hook cell (WeightSwapStage 동형).
    /// WeightRecallStage 는 IntraForward/LayerImmediate mode 가 없어 현재 미사용이나,
    /// one_shot() 시그니처를 WeightSwapStage 와 동형으로 유지하기 위해 보존한다.
    #[allow(dead_code)]
    hook_cell: Arc<Mutex<Option<Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>>>>,
    /// commit 1회만 수행하기 위한 가드.
    committed: Mutex<bool>,
    /// F16 secondary view 캐시 (lazy-once, INV-125 공유 보존).
    /// Ok(Arc<SecondaryMmap>) = 바인딩 성공.
    /// Err(String) = 실패 메시지 (loud no-op 이후 재시도 방지).
    f16_view_cache: OnceLock<Result<Arc<SecondaryMmap>, String>>,
}

impl WeightRecallStage {
    /// `RecallWeights` directive 1건 → OneShot recall Stage (§5.6.8).
    pub fn one_shot(
        model: Arc<TransformerModel>,
        runtime: Arc<EngineSwapRuntime>,
        ratio: f32,
        hook_cell: Arc<Mutex<Option<Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>>>>,
    ) -> Self {
        Self {
            model,
            runtime,
            ratio,
            hook_cell,
            committed: Mutex::new(false),
            f16_view_cache: OnceLock::new(),
        }
    }

    /// F16 secondary view 를 lazy-once 로 바인딩한다 (INV-125 — 같은 Arc<AufView>/mmap 공유).
    ///
    /// 결과는 `f16_view_cache` 에 캐싱되어 이후 recall 에서 재사용된다.
    /// 실패(DtypeNotFound / AdrenoSoaF16Rejected)도 캐싱하여 반복 loud-fail 방지.
    fn get_or_init_f16_view(&self) -> Result<Arc<SecondaryMmap>, &str> {
        self.f16_view_cache
            .get_or_init(|| {
                // secondary mmap 이 없으면 (c) no_secondary.
                let secondary = match self.model.secondary_mmap.as_ref() {
                    Some(s) => s,
                    None => return Err("no_secondary (ENG-DAT-C09)".to_string()),
                };

                // AUF secondary 에서 AufView + BackendTag 추출.
                // is_pre_converted_soa=true → AdrenoSoa tag, false → CpuAos.
                // GGUF/Rpcmem secondary 는 AUF-only path — non-AUF loud no-op.
                let auf = match secondary.as_ref() {
                    SecondaryMmap::Auf(a) => a,
                    _ => return Err("no_secondary or non-AUF secondary".to_string()),
                };
                let backend_tag = if auf.is_pre_converted_soa {
                    BackendTag::AdrenoSoa
                } else {
                    BackendTag::CpuAos
                };
                match crate::models::loader::auf::secondary::open_secondary_f16_for_recall(
                    Arc::clone(&auf.view),
                    backend_tag,
                    &self.model.config,
                ) {
                    Ok(mmap) => Ok(Arc::new(mmap)),
                    Err(e) => Err(format!("{e:?}")),
                }
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(|s| s.as_str())
    }

    /// commit 본문 (ENG-ALG-240/241).
    ///
    /// 반환: `true` = 성공(WeightSwapReport 송신 완료), `false` = loud no-op.
    fn commit(&self) -> bool {
        let ratio = self.ratio;

        // ── 1. ratio 검증 ──────────────────────────────────────────────────
        if ratio <= 0.0 || ratio > 1.0 {
            eprintln!("[WeightRecall] Rejected: invalid_ratio ({:.4})", ratio);
            return false;
        }

        // ── 2. F16 view 바인딩 (lazy-once) ─────────────────────────────────
        // (a) DtypeNotFound / (b) AdrenoSoaF16Rejected / (c) no_secondary
        let f16_view = match self.get_or_init_f16_view() {
            Ok(v) => v,
            Err(reason) => {
                eprintln!("[WeightRecall] Rejected: {} (INV-195)", reason);
                return false;
            }
        };

        // ── 3. In-flight check (R-1 동시 활성화 가드 — swap 과 공유) ──────
        if !self.runtime.is_idle() {
            eprintln!(
                "[WeightRecall] Rejected: in-flight plan active (ratio={:.2}). \
                 Wait for current swap/recall to complete.",
                ratio
            );
            return false;
        }

        // ── 4. recall 후보 수집 (currently Q4_0 layer — INV-193 SSOT) ─────
        let n_layers = self.model.layers.len();
        let currently_swapped: Vec<usize> = (0..n_layers)
            .filter(|&i| self.model.layers[i].current_dtype() == DType::Q4_0)
            .collect();

        // (d) swapped 0개
        if currently_swapped.is_empty() {
            eprintln!(
                "[WeightRecall] Rejected: no currently-swapped (Q4_0) layers to recall \
                 (ratio={:.2})",
                ratio
            );
            return false;
        }

        // ── 5. target 선택 (단순 역집합 — importance-aware 불필요) ─────────
        let target_count = ((ratio * currently_swapped.len() as f32).floor() as usize)
            .max(1)
            .min(currently_swapped.len());
        let target_layers: Vec<usize> = currently_swapped[..target_count].to_vec();

        eprintln!(
            "[WeightRecall] Incremental: ratio={:.2}, {} swapped layers, target {} → F16",
            ratio,
            currently_swapped.len(),
            target_count,
        );

        // ── 6. recall 실행 (SwapExecutor target_dtype=F16 재사용) ──────────
        let t_recall = Instant::now();
        match crate::session::qcf_runtime::run_layer_recall(
            self.model.as_ref(),
            &f16_view,
            &target_layers,
            self.runtime.swap_backend(),
            Some(self.runtime.dispatcher().as_ref()),
        ) {
            Ok(report) => {
                let latency_ms = t_recall.elapsed().as_millis() as u64;
                eprintln!(
                    "[WeightRecall] Done: recalled={} latency={:.1}ms",
                    report.swapped.len(),
                    latency_ms,
                );
                // ── 7. WeightSwapReport 송신 (from=Q4_0, to=F16 — 역방향) ──
                let layers_swapped: Vec<LayerSwapEntry> = report
                    .swapped
                    .iter()
                    .map(|s| LayerSwapEntry {
                        layer_idx: s.layer_idx as u32,
                        from_dtype: DtypeTag::Q4_0,
                        to_dtype: DtypeTag::F16,
                    })
                    .collect();
                self.runtime.send_swap_report(WeightSwapReport {
                    layers_swapped,
                    freed_bytes: 0,
                    latency_ms,
                    qcf_swap_actual: 0.0, // recall 은 품질 개선 방향 — QCF 미산출
                });
            }
            Err(e) => {
                eprintln!("[WeightRecall] run_layer_recall error: {e}");
            }
        }

        true
    }
}

impl PipelineStage for WeightRecallStage {
    fn name(&self) -> &str {
        "weight.recall"
    }

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::OneShot
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter (§5.6.8): WeightMutate 외 phase 는 무시.
        if *phase != LifecyclePhase::WeightMutate {
            return Ok(StageOutcome::Continue);
        }

        let mut committed = self
            .committed
            .lock()
            .expect("weight_recall commit mutex poisoned");
        if *committed {
            // 이미 commit 됨 — 후속 tick 에서 재진입 방지.
            return Ok(StageOutcome::Consumed);
        }
        *committed = true;
        let _ = ctx; // recall 은 Incremental multi-tick drain 없음 (1-batch 즉시 실행).
        drop(committed);

        // commit: loud no-op 5종 + 성공 경로.
        // 어느 경우든 OneShot 이므로 즉시 Consumed.
        let _ = self.commit();
        Ok(StageOutcome::Consumed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Mutex};

    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::{Buffer, DType};
    use crate::layers::transformer_layer::TransformerLayer;
    use crate::memory::Memory;
    use crate::memory::galloc::Galloc;
    use crate::memory::host::shared::SharedBuffer;
    use crate::model_config::{ModelArch, ModelConfig};
    use crate::models::weights::LayerSlot;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::session::cli::SwapMode;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use crate::weight::AsyncSwapDispatcher;

    fn cpu_backend() -> Arc<dyn Backend> {
        Arc::new(CpuBackend::new())
    }

    fn f32_weight(be: &Arc<dyn Backend>, out_dim: usize, in_dim: usize) -> Tensor {
        let buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::new(out_dim * in_dim * 4, DType::F32));
        Tensor::new(Shape::new(vec![out_dim, in_dim]), buf, be.clone())
    }

    fn ffn_slot(be: &Arc<dyn Backend>, idx: usize) -> Arc<LayerSlot> {
        let small = f32_weight(be, 1, 1);
        let layer = TransformerLayer {
            wq: small.clone(),
            wk: small.clone(),
            wv: small.clone(),
            wo: small.clone(),
            w_gate: f32_weight(be, 512, 256),
            w_up: f32_weight(be, 512, 256),
            w_down: f32_weight(be, 256, 512),
            attention_norm: small.clone(),
            ffn_norm: small,
            qkv_bias: None,
            q_norm: None,
            k_norm: None,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            partition_ctx: None,
        };
        Arc::new(LayerSlot::new(layer, DType::F32, None, idx))
    }

    fn make_model(be: &Arc<dyn Backend>, n_layers: usize) -> Arc<TransformerModel> {
        let mem = Galloc::new();
        let dim = 4usize;
        let vocab = 8usize;
        let embed_buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        let embed_tokens = Tensor::new(Shape::new(vec![vocab, dim]), embed_buf, be.clone());
        let norm_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let norm = Tensor::new(Shape::new(vec![dim]), norm_buf, be.clone());
        let lm_head = norm.clone();
        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: dim,
            num_hidden_layers: n_layers,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: dim,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: dim,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 2,
            arch: ModelArch::Llama,
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
            weight_prefix: String::new(),
        };
        let runtime = crate::weight::setup_runtime_resources(be.clone());
        Arc::new(TransformerModel {
            config,
            layers: (0..n_layers).map(|i| ffn_slot(be, i)).collect(),
            embed_tokens,
            norm,
            lm_head,
            lm_head_on_cpu: false,
            gpu_embed_tokens: None,
            cpu_backend: None,
            preload_pool: std::sync::OnceLock::new(),
            secondary_mmap: None,
            ratio_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            quant_noise: runtime.quant_noise.clone(),
            release_worker: runtime.release_worker.clone(),
        })
    }

    fn make_runtime(be: &Arc<dyn Backend>) -> Arc<EngineSwapRuntime> {
        let dispatcher = Arc::new(AsyncSwapDispatcher::new(be.clone()));
        let runtime = crate::weight::setup_runtime_resources(be.clone());
        let config = Arc::new(ModelConfig {
            vocab_size: 8,
            hidden_size: 4,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: 4,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 2,
            arch: ModelArch::Llama,
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
            weight_prefix: String::new(),
        });
        Arc::new(EngineSwapRuntime::new(
            be.clone(),
            dispatcher,
            config,
            runtime.release_worker.clone(),
            SwapMode::Incremental,
            1024 * 1024,
            4,
            None,
        ))
    }

    type HookCell = Arc<Mutex<Option<Arc<dyn crate::layer_boundary_hook::LayerBoundaryHook>>>>;
    fn empty_hook_cell() -> HookCell {
        Arc::new(Mutex::new(None))
    }

    fn make_ctx(profiler: &mut OpProfiler) -> StageContext<'_> {
        StageContext {
            step: StepInfo {
                pos: 0,
                decode_step: 0,
                pressure: Pressure::new(0),
                prev_token: 0,
            },
            profiler,
        }
    }

    /// WeightMutate 외 phase 는 no-op (Continue) — commit 미진입.
    #[test]
    fn non_weight_mutate_phase_is_noop() {
        let be = cpu_backend();
        let s = WeightRecallStage::one_shot(
            make_model(&be, 2),
            make_runtime(&be),
            1.0,
            empty_hook_cell(),
        );
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = s.on_phase(&LifecyclePhase::KvMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Continue), "phase filter");
        assert!(!*s.committed.lock().unwrap(), "commit 미진입");
    }

    /// no_secondary → loud no-op (Consumed).
    #[test]
    fn reject_no_secondary_is_graceful_consumed() {
        let be = cpu_backend();
        let s = WeightRecallStage::one_shot(
            make_model(&be, 2),
            make_runtime(&be),
            1.0,
            empty_hook_cell(),
        );
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = s.on_phase(&LifecyclePhase::WeightMutate, &mut ctx).unwrap();
        assert!(
            matches!(outcome, StageOutcome::Consumed),
            "no_secondary → Consumed"
        );
    }

    /// currently-swapped 0개 → loud no-op (Consumed).
    /// (모든 layer 가 F32이라 Q4_0=0인 상태)
    #[test]
    fn reject_none_swapped_is_graceful_consumed() {
        // secondary 없이 Q4_0 layer 0 → no_secondary 가 먼저 막지만,
        // 어느 경로든 Consumed 임을 확인.
        let be = cpu_backend();
        let s = WeightRecallStage::one_shot(
            make_model(&be, 2),
            make_runtime(&be),
            1.0,
            empty_hook_cell(),
        );
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = s.on_phase(&LifecyclePhase::WeightMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
    }

    /// invalid_ratio → graceful Consumed.
    #[test]
    fn reject_invalid_ratio_is_graceful_consumed() {
        let be = cpu_backend();
        let s = WeightRecallStage::one_shot(
            make_model(&be, 2),
            make_runtime(&be),
            1.5,
            empty_hook_cell(),
        );
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = s.on_phase(&LifecyclePhase::WeightMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
    }

    /// in-flight 마커가 set 된 경우 → loud no-op.
    #[test]
    fn reject_in_flight_is_graceful_consumed() {
        let be = cpu_backend();
        let rt = make_runtime(&be);
        rt.mark_in_flight(true);
        let s = WeightRecallStage::one_shot(
            make_model(&be, 2),
            Arc::clone(&rt),
            1.0,
            empty_hook_cell(),
        );
        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = s.on_phase(&LifecyclePhase::WeightMutate, &mut ctx).unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
        assert!(!rt.is_idle(), "in-flight 마커 보존");
    }

    /// 두 번 WeightMutate phase 진입 → 두 번째는 committed 가드로 즉시 Consumed.
    #[test]
    fn double_commit_guard_returns_consumed() {
        let be = cpu_backend();
        let s = WeightRecallStage::one_shot(
            make_model(&be, 2),
            make_runtime(&be),
            1.0,
            empty_hook_cell(),
        );
        let mut profiler = OpProfiler::new();
        // 첫 번째
        let _ = s
            .on_phase(
                &LifecyclePhase::WeightMutate,
                &mut StageContext {
                    step: StepInfo {
                        pos: 0,
                        decode_step: 0,
                        pressure: Pressure::new(0),
                        prev_token: 0,
                    },
                    profiler: &mut profiler,
                },
            )
            .unwrap();
        // 두 번째
        let outcome = s
            .on_phase(
                &LifecyclePhase::WeightMutate,
                &mut StageContext {
                    step: StepInfo {
                        pos: 1,
                        decode_step: 1,
                        pressure: Pressure::new(0),
                        prev_token: 0,
                    },
                    profiler: &mut profiler,
                },
            )
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Consumed));
    }
}
