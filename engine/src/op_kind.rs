//! Op-level identifier — L2 shared identifier (§13.8-G).
//!
//! Definitional ownership is ambiguous: `observability/profile/op_trace` is the
//! data producer, and L3 inference (`models/transformer.rs`,
//! `layers/transformer_layer/forward_gen.rs`) is the op-span consumer.
//! Both use the enum equally, so §13.8-G mandates L2 promotion.
//!
//! This file is the source-of-truth. `observability/profile/op_trace.rs`
//! maintains a BC re-export so existing `crate::observability::profile::op_trace::OpKind`
//! paths keep compiling.

/// Op buckets tracked by the tracer. Order matches the `forward_gen` flow,
/// followed by buckets for ops outside `forward_gen` but inside
/// `forward_into` (embedding, final norm, lm_head).
#[repr(usize)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OpKind {
    RmsNormAttn = 0,
    MatmulQkv = 1,
    Rope = 2,
    KvUpdate = 3,
    Attention = 4,
    MatmulWo = 5,
    RmsNormFfn = 6,
    MatmulFfnGateUp = 7,
    SiluMul = 8,
    MatmulFfnDown = 9,
    AddAssign = 10,
    /// Token embedding lookup + optional embed scaling (`forward_into`).
    Embedding = 11,
    /// Final RMSNorm before `lm_head` (`forward_into`).
    FinalNorm = 12,
    /// Final `lm_head` matmul (vocab × hidden). Major suspect in Sprint E.
    LmHead = 13,
}
