#[path = "spec/helpers.rs"]
mod helpers;
#[path = "spec/test_fsm_operating_mode.rs"]
mod test_fsm_operating_mode;
#[path = "spec/test_inv_003.rs"]
mod test_inv_003;
#[path = "spec/test_inv_004_017.rs"]
mod test_inv_004_017;
#[path = "spec/test_inv_005_006.rs"]
mod test_inv_005_006;
#[path = "spec/test_inv_015.rs"]
mod test_inv_015;
#[path = "spec/test_inv_020_026.rs"]
mod test_inv_020_026;
#[path = "spec/test_inv_062_064.rs"]
mod test_inv_062_064;
#[path = "spec/test_inv_072_076.rs"]
mod test_inv_072_076;
#[path = "spec/test_inv_081_082.rs"]
mod test_inv_081_082;

// ── 비-INV Spec 테스트 ──
#[path = "spec/test_eng_alg_010_012.rs"]
mod test_eng_alg_010_012;
#[path = "spec/test_eng_alg_020_022.rs"]
mod test_eng_alg_020_022;
#[path = "spec/test_eng_alg_030_048.rs"]
mod test_eng_alg_030_048;
#[path = "spec/test_eng_alg_051.rs"]
mod test_eng_alg_051;
#[path = "spec/test_eng_alg_060_092.rs"]
mod test_eng_alg_060_092;
#[path = "spec/test_eng_dat_012_031.rs"]
mod test_eng_dat_012_031;
#[path = "spec/test_eng_st_010_035.rs"]
mod test_eng_st_010_035;
#[path = "spec/test_eng_st_052_060.rs"]
mod test_eng_st_052_060;
#[path = "spec/test_proto_010_062.rs"]
mod test_proto_010_062;

// ── Protocol / State / Cross-cutting 테스트 ──
#[path = "spec/test_cross_060_061.rs"]
mod test_cross_060_061;
#[path = "spec/test_eng_st_032.rs"]
mod test_eng_st_032;
#[path = "spec/test_proto_042_073.rs"]
mod test_proto_042_073;

// ── Data / Protocol path 테스트 ──
#[path = "spec/test_eng_dat_c05_streaming.rs"]
mod test_eng_dat_c05_streaming;
#[path = "spec/test_eng_dat_c06_d2o.rs"]
mod test_eng_dat_c06_d2o;

// ── MSG-060 Engine Self-Utilization (2026-04 Phase 1) ──
// MSG-060 필드 17~18, MSG-067~069, INV-091~092
#[path = "spec/test_msg_060_self_util.rs"]
mod test_msg_060_self_util;

// ── Tensor Partition × Plan 통합 테스트 ──
#[path = "spec/test_eng_alg_200_plan_partition.rs"]
mod test_eng_alg_200_plan_partition;
#[path = "spec/test_inv_120_plan_partition_stale.rs"]
mod test_inv_120_plan_partition_stale;
#[path = "spec/test_partition_split_backend_retag.rs"]
mod test_partition_split_backend_retag;

// ── Weight Swap Phase 1/2 infrastructure (ENG-DAT-092/094, ENG-ALG-210/211) ──
// Runtime dynamic weight swap groundwork. Phase 1 exercises static shape +
// `LayerSlot` atomic primitives; Phase 2 swap execution and handler pipeline.
// Note: ENG-DAT-093 (TransformerWeights container) was removed in Stage 2
// cleanup — replaced by flat fields directly on `TransformerModel`.
#[path = "spec/test_eng_alg_210_initial_load.rs"]
mod test_eng_alg_210_initial_load;
#[path = "spec/test_eng_alg_211_weight_swap_handler.rs"]
mod test_eng_alg_211_weight_swap_handler;
#[path = "spec/test_eng_dat_092_layer_slot.rs"]
mod test_eng_dat_092_layer_slot;
#[path = "spec/test_eng_dat_094_secondary_mmap.rs"]
mod test_eng_dat_094_secondary_mmap;
#[path = "spec/test_inv_121_dynamic.rs"]
mod test_inv_121_dynamic;
#[path = "spec/test_inv_123_dynamic.rs"]
mod test_inv_123_dynamic;
#[path = "spec/test_inv_124_slot_dtype_consistency.rs"]
mod test_inv_124_slot_dtype_consistency;
#[path = "spec/test_inv_125_secondary_mmap_lifetime.rs"]
mod test_inv_125_secondary_mmap_lifetime;

// ── Weight Swap Phase 3.5 — ENG-ALG-219 global plan invalidation ──
// FullKernelPlan::execute() global ratio_generation stale check (INV-129).
#[path = "spec/test_eng_alg_219_plan_invalidation.rs"]
mod test_eng_alg_219_plan_invalidation;

// ── Weight Swap Phase 3.6 — ENG-ALG-221 noshuffle SOA registry coherence ──
// SwapExecutor invalidates OpenCLBackend noshuffle_soa_registry before the
// ratio_generation bump so FullKernelPlan rebuild re-registers against new
// cl_mem keys (INV-130).
#[path = "spec/test_inv_130_noshuffle_soa_coherence.rs"]
mod test_inv_130_noshuffle_soa_coherence;

// ── Weight Swap Phase 3.7a — SOA Re-conversion safety net (INV-131) ──
// ENG-ALG-222: clear → convert → register → ratio_generation bump 순서.
// INV-131: swap 후 첫 GPU matmul 직전 새 cl_mem이 registry에 등록 완료.
#[path = "spec/test_inv_131_soa_reconversion.rs"]
mod test_inv_131_soa_reconversion;

// ── Weight Swap Phase 3.7b — AUF v0.1 포맷 (INV-132~134, ENG-DAT-096, ENG-ALG-223) ──
// AUF reader fail-fast (INV-132), required section (INV-133),
// section offset/size 무결성 (INV-134), writer/stripper round-trip.
#[path = "spec/test_inv_132_134_auf.rs"]
mod test_inv_132_134_auf;

// ── ENG-DAT-096 AUF Secondary Mmap adaptor (WSWAP-3.7B-ENGINE) ──
// AUF 분기 detection, is_pre_converted_soa 계약, tensor bytes 서빙,
// SOA bypass 메커니즘 검증.
#[path = "spec/test_eng_dat_096_auf_secondary.rs"]
mod test_eng_dat_096_auf_secondary;

// ── AUF end-to-end shape propagation sanity (ENG-ALG-223 / ENG-DAT-096) ──
// auf_tool build → TensorEntry.shape 채움 → SecondaryMmap → swap_executor
// shape 검증 통과 안전망. 디바이스 측정 전 호스트에서 동일 차단 조건 사전 발견.
#[path = "spec/test_auf_e2e_sanity.rs"]
mod test_auf_e2e_sanity;

// ── Sprint G-1-D — INV-135/136 lm_head AUF load-path ──
// AUF lm_head Q4_0 entry: bit2=0→None(INV-136), bit2=1+ok→Some, 위반→Err.
// SecondaryMmap::as_auf_view() accessor + round-trip.
#[path = "spec/test_inv_135_136_lm_head_auf.rs"]
mod test_inv_135_136_lm_head_auf;

// ── AUF v0.2 Sprint B — INV-137~139 multi-dtype variant ──
// INV-137: multi-dtype shape 일치 의무 + lm_head 포함.
// INV-138: default_dtype 의무 + writer 안정 정렬 + dtype selection precedence.
// INV-139: capability bit 3 의미 + v0.1.x ↔ v0.2 양방향 호환.
#[path = "spec/test_inv_137_multi_dtype.rs"]
mod test_inv_137_multi_dtype;
#[path = "spec/test_inv_138_default_dtype.rs"]
mod test_inv_138_default_dtype;
#[path = "spec/test_inv_139_capability_bit3.rs"]
mod test_inv_139_capability_bit3;

// ── Weight Swap Phase 6.5 — ENG-ALG-226 fused SOA convert+transpose ──
// INV-140: fused single-dispatch kernel output is byte-equal to the legacy
// 4-step host-transpose path. Host-side comparison (skipped when no OpenCL
// platform or fused kernel fails to compile).
#[path = "spec/test_inv_140_fused_convert_byte_equal.rs"]
mod test_inv_140_fused_convert_byte_equal;

// ── AUF v0.2 Sprint C — ENG-ALG-224 multi-dtype writer ──
// dequant→requant 결정성, INV-138 sort, capability bit 3 자동 활성화.
#[path = "spec/test_eng_alg_224_writer_multi_dtype.rs"]
mod test_eng_alg_224_writer_multi_dtype;

// ── AUF v0.2 Sprint D — ENG-ALG-225 reader dtype dispatch ──
// reader lookup_tensor precedence (명시 > default_dtype > first-match).
// Adreno SOA × F16 reject, 단방향 swap 정합성, SwapExecutor 시그니처 unchanged.
#[path = "spec/test_eng_alg_225_reader_dispatch.rs"]
mod test_eng_alg_225_reader_dispatch;

// ── AUF v0.2 Sprint F — ISSUE-E-1 hotfix 회귀 격리 ──
// multi-dtype writer가 1-D tensor (RMSNorm 등)에 대해 Q4_0 변환을 적용하지 않도록
// build_dtype_candidates가 src_dtype 1개만 반환하는지 검증. F16 primary +
// `--secondary-dtype q4_0` swap 시 첫 token EOS 깨짐 회귀 방지.
#[path = "spec/test_issue_e_1_multi_dtype_byte_path.rs"]
mod test_issue_e_1_multi_dtype_byte_path;

// ── Sprint G-1-E — AUF lm_head Q4_0 통합 정확성 검증 ──
// 의문 1: ADRENO_SOA payload = q_buf||d_buf (SOA) layout 정합성.
// 의문 2: payload.bytes.len() == N*18 (SOA 총 크기 = AOS 총 크기 불변).
// Roundtrip, 후방 호환(INV-136), 숫자 정확성, SOA split 정합성.
#[path = "spec/test_g1e_lm_head_integration.rs"]
mod test_g1e_lm_head_integration;

// ── Weight Swap INV-122 v2 — Mixed Precision Forward 정확성 ──
// cond1: NMSE ≤ 0.01 (F16 baseline 대비), cond2: Δ top-1 ≤ 1 pp (Q4_0 baseline 대비).
// 환경변수 LLM_RS_TEST_MODEL_F16 / LLM_RS_TEST_MODEL_Q4 미설정 시 graceful skip.
#[path = "spec/test_inv_122_mixed_precision.rs"]
mod test_inv_122_mixed_precision;

// ── Weight Swap Phase 3 invariants (INV-126/127/128, WSWAP-3-TEST) ──
// Stage C: LuaPolicy integration + Phase 3 invariant spec tests.
#[path = "spec/test_inv_126_dtype_reserved.rs"]
mod test_inv_126_dtype_reserved;
#[path = "spec/test_inv_127_nan_epsilon.rs"]
mod test_inv_127_nan_epsilon;
#[path = "spec/test_inv_128_collector_leak_guard.rs"]
mod test_inv_128_collector_leak_guard;
#[path = "spec/test_wswap_e2e_phase3.rs"]
mod test_wswap_e2e_phase3;

// ── Layer-swap QCF dump 인프라 (zazzy-herding-bonbon Phase 1) ──
// dump_qcf_swap_json + QcfSwapDumpContext JSON schema 계약 검증.
// 외부 harness(pact2026/experiments/scripts/)가 소비하는 JSON 포맷 안정성 보장.
#[path = "spec/test_qcf_swap_dump.rs"]
mod test_qcf_swap_dump;

// ── SEQ 통합 테스트 ──
#[path = "spec/test_seq_020_035.rs"]
mod test_seq_020_035;
#[path = "spec/test_seq_040_064.rs"]
mod test_seq_040_064;
#[path = "spec/test_seq_070_093.rs"]
mod test_seq_070_093;
#[path = "spec/test_seq_095_098.rs"]
mod test_seq_095_098;

// ── ARGUS QCF 실험 인프라 spec 테스트 (Steps 1–6) ──
// aggregation/retention/entropy invariant + β-amplified CAOTE 회귀 가드 +
// β=1.0 fast path 비부식성 검증.
#[path = "spec/test_qcf_backward_compat.rs"]
mod test_qcf_backward_compat;
#[path = "spec/test_qcf_beta_regression.rs"]
mod test_qcf_beta_regression;
#[path = "spec/test_qcf_experimental.rs"]
mod test_qcf_experimental;

// ── INV-141 PrimaryReleaseWorker drain contract (ENG-ALG-228 / ENG-DAT-100) ──
#[path = "spec/test_inv_141_release_worker_drain.rs"]
mod test_inv_141_release_worker_drain;

// ── INV-143 BorrowedMmapBuffer mmap lifetime (ENG-ALG-227) ──
// AOS 무변환 경로에서 borrow buffer가 secondary Arc clone을 보관하여
// mmap region이 copy_weight_from/copy_from 호출 중 drop되지 않음을 보증.
#[path = "spec/test_inv_143_borrow_buffer_lifetime.rs"]
mod test_inv_143_borrow_buffer_lifetime;

// ── ENG-ALG-229 Targeted prefault for swap target layers ──
// prefault_layers(target_layers)는 swap 대상 layer의 byte range만 page-touch.
// ratio < 1.0 swap batch에서 비대상 layer 페이지 접근 ~40 ms 절감.
#[path = "spec/test_eng_alg_229_targeted_prefault.rs"]
mod test_eng_alg_229_targeted_prefault;

// ── WSWAP-6-PREFAULT Eager prefault at model load ──
// SecondaryMmap::prefault() at model load time to eliminate per-swap cold page
// faults (~328 ms on Galaxy S25). CLI: --eager-prefault-secondary.
// Scenarios: valid secondary (no-panic + data intact), idempotent double-call,
// silent-skip when no secondary is configured.
#[path = "spec/test_eng_alg_232_eager_prefault.rs"]
mod test_eng_alg_232_eager_prefault;

// ── ENG-ALG-230/231 + INV-142: async write_buffer + stage gate ordering ──
// alloc_and_upload_soa_buffers blocking=false (ENG-ALG-230);
// single backend.synchronize() gate before invalidate/restore/bump (ENG-ALG-231).
// INV-142: queue idle before ratio_generation bump.
#[path = "spec/test_inv_142_stage_gate_ordering.rs"]
mod test_inv_142_stage_gate_ordering;

// ── Layer-Incremental Swap Stage 1 MVP (LISWAP-1, ENG-ALG-232~234, INV-144~146) ──
// IncrementalSwapPlan data structure (ENG-ALG-232, INV-145),
// decode loop dispatch simulation (ENG-ALG-233, INV-146),
// forward snapshot consistency under incremental swap (INV-144),
// drain monotone property (INV-145),
// tick/batch equivalence (INV-146).
#[path = "spec/test_eng_alg_232_incremental_plan.rs"]
mod test_eng_alg_232_incremental_plan;
#[path = "spec/test_eng_alg_233_main_loop_dispatch.rs"]
mod test_eng_alg_233_main_loop_dispatch;
#[path = "spec/test_inv_144_forward_snapshot.rs"]
mod test_inv_144_forward_snapshot;
#[path = "spec/test_inv_145_drain_monotone.rs"]
mod test_inv_145_drain_monotone;
#[path = "spec/test_inv_146_tick_batch_equivalence.rs"]
mod test_inv_146_tick_batch_equivalence;

// ── LISWAP-2 Phase 3 — SwapExecutor async dispatch path (prototype) ──
// `execute_on_slots`에 `async_dispatcher: Option<&AsyncSwapDispatcher>` 추가.
// async path: `enqueue_write_async` + per-event `wait_event_blocking` (INV-142 대체).
// sync path: backward-compat (기존 `backend.synchronize()` gate 유지).
#[path = "spec/test_async_swap_executor.rs"]
mod test_async_swap_executor;

// ── LISWAP-3 — HostPtrPool slot lifecycle (Direction A, Stage 3 prototype) ──
// `CL_MEM_ALLOC_HOST_PTR` 슬롯 풀 생성/획득/해제 + fill round-trip.
// Plan: compiled-chasing-hopper.md Direction A 트랙. spec ID 미발급.
#[path = "spec/test_host_ptr_pool.rs"]
mod test_host_ptr_pool;

// ── LISWAP-4 — Intra-forward Layer-aligned Swap (ENG-ALG-235~238, INV-147~150) ──
// INV-147: hook=None forward path zero overhead (NoOpHook microbench, host).
// INV-148: IntraForwardSwapPlan dispatch idempotency.
// INV-149: wait gate ordering + pending_event registry semantics.
// INV-150: plan run-to-completion finalize (drain → sync → bump → invalidate).
#[path = "spec/test_inv_147_hook_zero_overhead.rs"]
mod test_inv_147_hook_zero_overhead;
#[path = "spec/test_inv_148_plan_dispatch_idempotent.rs"]
mod test_inv_148_plan_dispatch_idempotent;
#[path = "spec/test_inv_149_wait_gate_ordering.rs"]
mod test_inv_149_wait_gate_ordering;
#[path = "spec/test_inv_150_plan_run_to_completion.rs"]
mod test_inv_150_plan_run_to_completion;

// ── INV-LAYER-006 (DecodeLoop field abstraction) +
//    INV-LAYER-007 (DecodeLoopBuilder typestate, trybuild) ──
// Phase 4-2: 6-trait + DecodeLoopBuilder typestate API. INV-LAYER-006 is a
// source-grep on session/decode_loop.rs; INV-LAYER-007 uses trybuild to assert
// that `DecodeLoopBuilder::new().build()` fails to compile and that a minimal
// Forward impl (prefill+step only) compiles.
#[path = "spec/test_inv_layer_006.rs"]
mod test_inv_layer_006;
#[path = "spec/test_inv_layer_007.rs"]
mod test_inv_layer_007;

// ── INV-DECODE-STAGE-004~007: driver↔Stage 계약 + PipelineRegistry 의미론 ──
// Phase β-1 계약 확정 (2026-06-10). §5.2.1 4건:
// 001: KV-PHASE mutation 금지 — PreLayer/PostLayer/Fine β 미발화 orphan 검증.
// 002: 폐기 tombstone (2026-05-28, KvBundle trait 폐기).
// 003: 폐기 tombstone (2026-05-28, β sync 모델 채택).
#[path = "spec/test_inv_decode_stage_001_002_003.rs"]
mod test_inv_decode_stage_001_002_003;

// 004: on_phase Result 의미론 (Continue/Consumed/Stop/Err).
// 005: submit 순서 = 순회 순서 + EvictionStage phase 공유.
// 006: StageContext 2-field 슬림 컴파일 강제.
// 007: OneShot GC 정확히 1회 (자기 phase 도달 전 no-GC).
#[path = "spec/test_inv_decode_stage_004_005_006_007.rs"]
mod test_inv_decode_stage_004_005_006_007;

// ── Phase 4-5-d: ChatSession multi-turn + /reset + ensure_capacity + stats_line ──
// G2: multi-turn KV pos 누적 보존 (DecodeLoop turn 사이 owned 재사용)
// G3: /reset — pos=0, evicted_total=0, Forward::reset_kv 호출
// G4: ensure_capacity — Standard/Kivi/Offload 모드별 overflow bail
// G1/D5: stats_line 포맷 byte-identical
#[path = "spec/test_chat_session_multi_turn.rs"]
mod test_chat_session_multi_turn;

// ── Phase 4-5-e: run_chat_repl_v2 multi-turn pos 누적 + first token bit-identical ──
// G2-REPL: turn 2 prefill pos 누적 (pos += tokens.len()), bit-identical 검증
// G2-REPL-2: 결정론적 first token (DetForward + GreedySampler)
// G2-REPL-3: reset 후 pos=0에서 재시작
#[path = "spec/test_chat_repl_v2_multi_turn.rs"]
mod test_chat_repl_v2_multi_turn;

// ── --kv-mode (FORMAT-axis mode/knob declaration) ──
// `--kv-mode` String parsing + KvModeArgs flatten; resolves against the engine
// KV-mode registry (no closed ValueEnum). `--kv-mode kivi` still parses + resolves.
#[path = "spec/test_kv_mode_args.rs"]
mod test_kv_mode_args;

// ── Session prefix KV cache (INV-189~191, ENG-085) ──
// ENG-080~085 capability + snapshot timing + invalidation + byte-identical + CLI
#[path = "spec/test_eng_085_prefix_cache_cli.rs"]
mod test_eng_085_prefix_cache_cli;
#[path = "spec/test_inv_189_snapshot_timing.rs"]
mod test_inv_189_snapshot_timing;
#[path = "spec/test_inv_190_invalidation.rs"]
mod test_inv_190_invalidation;
#[path = "spec/test_inv_191_restore_byte_identical.rs"]
mod test_inv_191_restore_byte_identical;

// ── INV-RPCMEM-001~008: RpcmemAllocator backend-agnostic 분리 (Sprint 2a Phase 2) ──
// ENG-RPCMEM-010~042 + ENG-RPCMEM-C01~C04 구현 검증.
// Android-only INV (001/002/005/007) 는 호스트에서 source-grep / skip.
// 호스트 검증 가능 INV (003/004/006/007/008) 는 mock allocator 또는 source-grep.
#[path = "spec/test_inv_rpcmem_001_android_only.rs"]
mod test_inv_rpcmem_001_android_only;
#[path = "spec/test_inv_rpcmem_002_single_instance.rs"]
mod test_inv_rpcmem_002_single_instance;
#[path = "spec/test_inv_rpcmem_003_per_buffer_fallback.rs"]
mod test_inv_rpcmem_003_per_buffer_fallback;
#[path = "spec/test_inv_rpcmem_004_no_qnn_dlopen.rs"]
mod test_inv_rpcmem_004_no_qnn_dlopen;
#[path = "spec/test_inv_rpcmem_005_drop_order.rs"]
mod test_inv_rpcmem_005_drop_order;
#[path = "spec/test_inv_rpcmem_006_cli_mutex.rs"]
mod test_inv_rpcmem_006_cli_mutex;
#[path = "spec/test_inv_rpcmem_007_activation_no_rpcmem.rs"]
mod test_inv_rpcmem_007_activation_no_rpcmem;
#[path = "spec/test_inv_rpcmem_008_no_raw_clientbuf.rs"]
mod test_inv_rpcmem_008_no_raw_clientbuf;

// ── INV-192~195: RecallWeights 메커니즘 (§5.6.8, ENG-ALG-240/241, MSG-043) ──
// INV-192: explicit trigger only (RestoreDefaults 발화 안 함).
// INV-193: currently-Q4_0 layer 만 후보 + ratio 선택 로직.
// INV-194: recall 후 NMSE ≤ 0.01 정확성 (device-gated, AUF 모델 필요).
// INV-195: loud no-op 5종 + double_commit_guard (panic/Err-강하 금지).
#[path = "spec/test_inv_192_recall_explicit_trigger.rs"]
mod test_inv_192_recall_explicit_trigger;
#[path = "spec/test_inv_193_recall_candidate_selection.rs"]
mod test_inv_193_recall_candidate_selection;
#[path = "spec/test_inv_194_recall_accuracy.rs"]
mod test_inv_194_recall_accuracy;
#[path = "spec/test_inv_195_recall_loud_noop.rs"]
mod test_inv_195_recall_loud_noop;
