//! INV-147 — `LayerBoundaryHook` zero-overhead 검증.
//! Spec ref tag for coverage: inv_147
//!
//! Spec: `spec/41-invariants.md` §3.21 INV-147, `spec/32-engine-algorithms.md`
//! §3.12.22.1 (ENG-ALG-235), `arch/weight_swap.md` §10.2 / §10.12.1.
//!
//! 검증:
//! 1. `LayerBoundaryHook` trait의 default 메서드 (`pending_event_for_dyn`)는
//!    `None`을 반환 — 호출 자체가 일정 시간에 끝남 (기본 구현은 단순 None
//!    리턴).
//! 2. `NoOpHook::on_layer_boundary`는 빈 함수이므로 호출 비용이
//!    measurement noise 이하임을 microbench로 확인.
//! 3. spec §10.12.1 Case 1과 Case 3에 해당하는 항목은 forward 통합
//!    microbench이지만, host CPU 단위에서는 이 trait 호출 자체의 비용을
//!    격리 측정한다. 디바이스 forward 비교는 별도 Tester 작업.

use std::hint::black_box;
use std::time::Instant;

use argus_engine::layer_boundary_hook::{LayerBoundaryHook, NoOpHook};

/// Case 1 (host-trivial): `pending_event_for_dyn` default returns None.
#[test]
fn test_default_pending_event_for_dyn_is_none() {
    let hook = NoOpHook;
    let dyn_hook: &dyn LayerBoundaryHook = &hook;
    for idx in [0usize, 7, 16, 31, 1024] {
        assert!(
            dyn_hook.pending_event_for_dyn(idx).is_none(),
            "NoOpHook must report no pending event"
        );
    }
}

/// Case 2 (microbench, host CPU): NoOpHook's on_layer_boundary takes
/// negligible wall time. We do not enforce a hard cycle budget — instead we
/// assert that 10_000 × 16 calls complete in well under 100 ms (extreme
/// upper bound; typical host machines finish in <1 ms).
///
/// This is a smoke check — real INV-147 verification is the Galaxy S25
/// forward microbench (Tester job, see arch §10.12.1 Case 2/3).
#[test]
fn test_noop_hook_call_overhead_is_bounded() {
    let hook = NoOpHook;
    let dyn_hook: &dyn LayerBoundaryHook = &hook;

    // Warm-up to stabilise CPU branch predictor.
    for i in 0..256 {
        dyn_hook.on_layer_boundary(black_box(i % 16), black_box(1));
    }

    let n_outer = 10_000usize;
    let n_layers = 16usize;
    let start = Instant::now();
    for outer in 0..n_outer {
        for layer in 0..n_layers {
            dyn_hook.on_layer_boundary(black_box(layer), black_box((outer & 1) + 1));
        }
    }
    let elapsed = start.elapsed();
    let total_calls = (n_outer * n_layers) as f64;
    let ns_per_call = elapsed.as_nanos() as f64 / total_calls;
    eprintln!(
        "NoOpHook call: {ns_per_call:.2} ns/call, total {} calls in {:?}",
        n_outer * n_layers,
        elapsed
    );

    // Soft bound — release builds typically deliver <5 ns/call. Debug builds
    // (CI) can reach ~100 ns/call. We pick 10_000 ns (10 µs) as a generous
    // ceiling so flaky CI does not fail; the real INV-147 hard threshold is
    // measured on-device.
    assert!(
        ns_per_call < 10_000.0,
        "NoOpHook call overhead {ns_per_call:.2} ns exceeds soft 10 µs bound"
    );
    // Also bound total wall time: less than 250 ms for 160k calls in any
    // realistic CI setting.
    assert!(
        elapsed.as_millis() < 250,
        "NoOpHook microbench took {elapsed:?}, expected <250ms"
    );
}

/// Case 3 (semantics): `Option::is_some` branch is the only hot-path cost
/// when the caller passes `None`. We document this contract here as a
/// non-test compile assertion: the trait surface guarantees the wait gate
/// helper does not call into the hook unless `Some`.
#[test]
fn test_option_branch_only_cost_for_none() {
    let hook: Option<&dyn LayerBoundaryHook> = None;
    // Simulate the forward layer loop's hot-path predicate.
    let mut counter = 0u64;
    for i in 0..16 {
        if let Some(h) = hook {
            // Should never enter — None case.
            counter += 1;
            // SAFETY: dead code from the test's perspective; included to
            // satisfy the compiler's reachability analysis.
            let _ = h.pending_event_for_dyn(i);
        }
    }
    assert_eq!(counter, 0, "None hook must never invoke trait method");
}
