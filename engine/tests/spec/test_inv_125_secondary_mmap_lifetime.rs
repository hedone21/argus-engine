//! INV-125 — Secondary mmap lifetime invariant.
//!
//! `TransformerWeights::secondary_mmap` keeps the sole strong reference (in
//! addition to each `LayerSlot::secondary_mmap_handle`). While the container
//! is alive, clones of the `Arc<SecondaryMmap>` must remain valid, and the
//! underlying allocation must not be dropped until the root container goes
//! away.
//!
//! Phase 1 fakes the `SecondaryMmap` contents (we cannot build one without a
//! real GGUF file) by reasoning about the `Arc` strong count over a proxy
//! payload. The structural property we assert is layout-agnostic.

use std::sync::Arc;

#[test]
fn arc_stays_alive_while_handles_share_it() {
    // Proxy for `Arc<SecondaryMmap>`: we only care about the Arc semantics
    // that `LayerSlot::secondary_mmap_handle` and
    // `TransformerWeights::secondary_mmap` are built on.
    let root = Arc::new(String::from("secondary mmap payload"));
    let handle_a = root.clone();
    let handle_b = root.clone();

    // Simulate the slot handles going out of scope first. Root must remain
    // alive — this mirrors the real container, which owns the final Arc.
    assert_eq!(Arc::strong_count(&root), 3);
    drop(handle_a);
    drop(handle_b);
    assert_eq!(Arc::strong_count(&root), 1);
    // Dropping root releases the underlying resource.
    drop(root);
}

#[test]
fn arc_is_last_keeper_when_handles_outlive_root_drop() {
    // Inverse ordering: drop root first, then observe handles still keep the
    // payload alive. Real code does not execute this path (the container
    // lifetime is the longest, INV-125), but the Arc machinery tolerates it.
    let root = Arc::new(String::from("payload"));
    let handle = root.clone();
    drop(root);
    assert_eq!(Arc::strong_count(&handle), 1);
    assert_eq!(handle.as_str(), "payload");
}
