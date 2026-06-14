//! INV-LAYER-007 negative: build() must not compile without a Forward.

use argus_engine::session::DecodeLoopBuilder;

fn main() {
    let _loop = DecodeLoopBuilder::new().build();
}
