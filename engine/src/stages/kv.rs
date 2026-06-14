//! KV-mutate Stage 거주지 (§2.1).
//!
//! 예정 입주자(Phase α-K): `EvictionStage` / `D2OStage` / `KviQuantizeStage` /
//! `SwapDispatchStage` / `TierMoveStage`(신규). 현 `pressure/{eviction,d2o,quantize,swap}_handler.rs`
//! 의 `handle()` 트리거 부분 + tier_move 신규. 알고리즘(d2o merge ~440 LOC·`offload_one`/
//! `recall_one`)은 `kv/`(현 `pressure/`)로 추출한다(함수 단위 cut, G3-reconcile Q3).
//!
//! `kv/` 정책·포맷에 수평 의존(L3→L3).
//!
//! **입주자 1호(Phase β-3 commit B)**: [`eviction::EvictionStage`] — `KvMutate` phase 에서
//! CacheManager UER(take/put_inner) 경유로 발화하는 v2 `PipelineStage`. v1 `try_evict`(AB-1)와
//! 산출 동일(madvise/CacheEvent/min-floor 회계 보존).
//!
//! **입주자 2호(AB-2 §5.7)**: [`kivi_quant::KiviQuantStage`] — `KvMutate` phase 에서 KIVI cache
//! bit-width 를 `transition_bits` 로 런타임 전환하는 OneShot Stage. EvictionStage 와 형제(KV 축).
//! 나머지 입주자(D2O/Swap/TierMove)는 후속 substep 에서 채운다.

pub mod eviction;
pub mod kivi_quant;
pub mod offload;
