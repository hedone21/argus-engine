//! L2 instrument helpers — §13.8-H instrument macro helper 정책.
//!
//! 매크로 정의는 L2, expansion 내부에서 cross-cutting concrete 참조.
//! cfg gate(`profile` feature)로 production 빌드 시 zero-cost 컴파일 제거.

/// Pattern A 매크로 — Timer + QCF counter RAII 측정.
///
/// 사용: `let _t = qcf_timer!(NLL);` → cfg=on이면 Timer::start(&NLL),
/// cfg=off이면 unit `()` (no-op).
///
/// `$counter`는 `observability::profile::quality_metrics` 모듈 내 static 식별자
/// (NLL, QCF_KV_UNIFIED, QCF_KV_DRYRUN, QCF_WEIGHT_SWAP, QCF_LAYER_SKIP, DECODE_TOTAL).
#[macro_export]
macro_rules! qcf_timer {
    ($counter:ident) => {{
        #[cfg(feature = "profile")]
        let _g = $crate::observability::profile::quality_metrics::Timer::start(
            &$crate::observability::profile::quality_metrics::$counter,
        );
        #[cfg(not(feature = "profile"))]
        let _g = ();
        _g
    }};
}

/// Pattern B 매크로 — op_trace span 시작 (`Option<Instant>` 반환).
///
/// 사용: `let tok = op_start!();` → cfg=on이면 `op_trace::start()`,
/// cfg=off이면 `None`. embedding/FinalNorm/LmHead 등 OpKind 인자가
/// 불필요한 (phase hook 미사용) 호출 지점에서 사용.
#[macro_export]
macro_rules! op_start {
    () => {{
        #[cfg(feature = "profile")]
        let _tok = $crate::observability::profile::op_trace::start();
        #[cfg(not(feature = "profile"))]
        let _tok: Option<std::time::Instant> = None;
        _tok
    }};
}

/// Pattern B 매크로 — OpKind와 함께 시작 (phase hook 활성 경로).
///
/// `op_trace::start_op(OpKind::$kind)`를 호출하여 PhaseHook이 등록된 경우
/// `on_op_start(kind)`를 fire. forward_gen 11개 op 측정에 사용.
#[macro_export]
macro_rules! op_start_kind {
    ($kind:ident) => {{
        #[cfg(feature = "profile")]
        let _tok =
            $crate::observability::profile::op_trace::start_op($crate::op_kind::OpKind::$kind);
        #[cfg(not(feature = "profile"))]
        let _tok: Option<std::time::Instant> = None;
        _tok
    }};
}

/// Pattern B 매크로 — span 종료 + 라벨링.
///
/// `$tok`: `op_start!` 또는 `op_start_kind!` 반환값.
/// `$kind`: `OpKind` variant (식별자).
/// `$backend`: `&Arc<dyn Backend>` (sync 호출용).
/// `$is_gpu`: bool (sync mode에서만 의미 있음).
#[macro_export]
macro_rules! op_record {
    ($tok:expr, $kind:ident, $backend:expr, $is_gpu:expr) => {{
        #[cfg(feature = "profile")]
        $crate::observability::profile::op_trace::record(
            $tok,
            $crate::op_kind::OpKind::$kind,
            $backend,
            $is_gpu,
        );
        #[cfg(not(feature = "profile"))]
        {
            // unused 경고 방지 (cfg=off 빌드에서도 평가 부수효과 보존).
            let _ = ($tok, $backend, $is_gpu);
        }
    }};
}

/// Pattern B 매크로 — forward call 카운터 증가.
///
/// `op_trace::note_forward_call()`를 호출. 단일 decode token당 1회.
#[macro_export]
macro_rules! op_note_forward_call {
    () => {{
        #[cfg(feature = "profile")]
        $crate::observability::profile::op_trace::note_forward_call();
    }};
}

// ── Pattern C: OpInstrument trait ─────────────────────────────────────────────
//
// Op-level profiler를 trait object로 erasure. `observability/profile/ops.rs`의
// `OpProfiler`/`PrefillOpProfiler`가 impl 제공.
//
// 설계 원칙:
//   - 두 struct의 *공통 mutate 경로*만 trait method로 노출.
//   - 한 쪽만 의미 있는 method는 default no-op으로 처리 (타 struct는 override 불필요).
//   - B-2d-1: trait + impl 정의만. 사용처(Option<&mut OpProfiler/PrefillOpProfiler>)
//     를 Option<&mut dyn OpInstrument>로 교체하는 작업은 B-2d-2.

/// Op-level profiler abstraction for decode/prefill forward passes.
///
/// `OpProfiler` (decode)와 `PrefillOpProfiler` (prefill) 양쪽이 impl.
/// `Send` bound: forward pass는 단일 스레드이지만 struct holder 경로에서 스레드
/// 경계를 넘을 수 있으므로 포함.
pub trait OpInstrument: Send {
    /// Named op의 microsecond 값을 누적한다.
    ///
    /// `op_name`은 `"rms_norm"`, `"matmul_qkv"` 등 known op label.
    /// 알 수 없는 label은 구현체 재량으로 무시하거나 `other` 버킷에 추가.
    fn record_op_us(&mut self, op_name: &'static str, elapsed_us: u64);

    /// GPU→CPU attention fallback 이벤트를 기록한다 (prefill 전용).
    ///
    /// `OpProfiler`는 default no-op. `PrefillOpProfiler`만 override.
    fn record_cpu_fallback(&mut self, _head_dim: usize, _dtype_str: &str, _reason: &'static str) {}

    /// 레이어 종료 시 layer_count를 증가시킨다 (prefill 전용).
    ///
    /// `OpProfiler`는 default no-op. `PrefillOpProfiler`만 override.
    fn record_layer_end(&mut self) {}

    /// GPU 프로파일 이벤트 맵을 merge한다 (decode 전용).
    ///
    /// `PrefillOpProfiler`는 default no-op. `OpProfiler`만 override.
    fn merge_events(&mut self, _events: &std::collections::HashMap<String, u64>) {}

    /// 토큰 1개 처리 완료를 기록한다 (decode 전용, count 증가).
    ///
    /// `PrefillOpProfiler`는 default no-op. `OpProfiler`만 override.
    fn note_token(&mut self) {}
}
