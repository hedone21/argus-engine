//! INV-148 — `IntraForwardSwapPlan` dispatch idempotency.
//! Spec ref tag for coverage: inv_148
//!
//! Spec: `spec/41-invariants.md` §3.21 INV-148, `spec/32-engine-algorithms.md`
//! §3.12.22.2 (ENG-ALG-236), `arch/weight_swap.md` §10.2 / §10.12.2.
//!
//! 검증: 단일 `IntraForwardSwapPlan` instance에서 동일 layer index `idx`는
//! 정확히 1회만 dispatch된다. `mark_dispatched(idx)` 후 `should_dispatch(idx)`는
//! 영구적으로 false. 동일 idx에 대한 중복 mark는 안전 (no-op).

use argus_engine::weight::IntraForwardSwapPlan;

#[test]
fn test_should_dispatch_then_mark_disables_redispatch() {
    let mut plan = IntraForwardSwapPlan::new(vec![3, 5, 7], 0);
    assert!(plan.should_dispatch(3), "idx in dispatch_at must dispatch");
    plan.mark_dispatched(3);
    assert!(
        !plan.should_dispatch(3),
        "after mark_dispatched, should_dispatch must be false"
    );
    assert!(
        plan.should_dispatch(5),
        "unmarked idx should still dispatch"
    );
}

#[test]
fn test_double_mark_is_safe() {
    let mut plan = IntraForwardSwapPlan::new(vec![3], 0);
    plan.mark_dispatched(3);
    plan.mark_dispatched(3); // double mark — must not panic
    assert!(!plan.should_dispatch(3));
    assert!(plan.is_complete());
}

#[test]
fn test_idx_outside_dispatch_at_never_dispatches() {
    let plan = IntraForwardSwapPlan::new(vec![3], 0);
    assert!(!plan.should_dispatch(0));
    assert!(!plan.should_dispatch(99));
    // marking such an idx should also leave is_complete intact (only depends
    // on dispatch_at superset).
}

#[test]
fn test_complete_after_marking_all() {
    let mut plan = IntraForwardSwapPlan::new(vec![3, 5, 7], 0);
    assert!(!plan.is_complete());
    for i in [3, 5, 7] {
        plan.mark_dispatched(i);
    }
    assert!(plan.is_complete());
    assert_eq!(plan.pending_layers().count(), 0);
}

#[test]
fn test_empty_plan_is_complete_immediately() {
    let plan = IntraForwardSwapPlan::new(vec![], 0);
    assert!(
        plan.is_complete(),
        "empty plan must be is_complete=true (BTreeSet::is_superset of empty set)"
    );
}

#[test]
fn test_pending_layers_diff() {
    let mut plan = IntraForwardSwapPlan::new(vec![1, 2, 3], 0);
    plan.mark_dispatched(2);
    let pending: Vec<usize> = plan.pending_layers().collect();
    assert_eq!(pending, vec![1, 3]);
}

#[test]
fn test_duplicate_input_layers_deduped() {
    // BTreeSet collapses dupes — INV-148 corollary: each layer appears at
    // most once in dispatch_at.
    let mut plan = IntraForwardSwapPlan::new(vec![3, 3, 3], 0);
    assert!(plan.should_dispatch(3));
    plan.mark_dispatched(3);
    assert!(plan.is_complete());
}

#[test]
fn test_started_at_token_preserved() {
    let plan = IntraForwardSwapPlan::new(vec![1, 2], 42);
    assert_eq!(plan.started_at_token(), 42);
}
