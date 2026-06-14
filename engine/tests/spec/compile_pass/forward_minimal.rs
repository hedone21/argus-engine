//! INV-LAYER-007 positive: implementing only Forward::prefill + step must
//! compile (lifecycle hooks are default no-op per arch §11 decision #2).

use argus_engine::session::{DecodeLoopBuilder, Forward, StepCtx};

struct Dummy;

impl Forward for Dummy {
    fn prefill(&mut self, _tokens: &[u32], _start_pos: usize) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0; 4])
    }
    fn step(&mut self, _ctx: &StepCtx, _token: u32) -> anyhow::Result<Vec<f32>> {
        Ok(vec![0.0; 4])
    }
}

fn main() {
    let _loop = DecodeLoopBuilder::new().with_forward(Dummy).build();
}
