//! Phase 4-3 C2 positive: `ModelForward` implements `Forward`, so a
//! `DecodeLoopBuilder` can accept it via `.with_forward(...)` and reach the
//! `.build()` typestate state.
//!
//! Compile-only fixture — no runtime model is loaded. The presence of the
//! `with_forward(model_forward).build()` chain is what we want trybuild to
//! type-check. `ModelForward::new` is not invoked because that would require
//! a backend / memory / model triple this fixture cannot construct.

use argus_engine::session::forward::ModelForward;
use argus_engine::session::{DecodeLoopBuilder, Forward};

fn _accepts_model_forward(mf: ModelForward) {
    let _ = DecodeLoopBuilder::new().with_forward(mf).build();
}

fn _accepts_via_trait_object(mf: Box<dyn Forward>) {
    let _: Box<dyn Forward> = mf;
}

fn main() {}
