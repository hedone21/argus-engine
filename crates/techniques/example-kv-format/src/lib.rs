//! Example format technique crate — the format-axis proof of "add a folder = zero
//! engine-core edits" + a contributor template. The format-axis counterpart of the
//! stage-axis [`example-keep-recent`](../example-keep-recent).
//!
//! This crate depends only on [`technique_api`], implements
//! [`KVFormat`](technique_api::KVFormat), and registers itself via the `register_kv_format!`
//! macro (static linkme + cdylib C-ABI dual-wiring). Its descriptor is q4_0-like (block_elems
//! 32 / bits 4 / PerBlockF16 / Nibble) — the engine has no matching `DType` variant, so it is
//! handled by the generic floor (opaque), the same class as `synth_q4`.
//!
//! GATE-C v2: `cargo build -p example-kv-format --features plugin-cdylib` produces the `.so` →
//! the host dlopens it zero-compile via `register_dynamic_formats`. **Because the engine does
//! not force-link this crate** (unlike synth_q4), it demonstrates the dynamic-registration
//! success path (`make_format` fallback) with no static collision.

use technique_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

/// Example format that provides only a q4_0-like descriptor (name + layout, 2 methods).
struct ExampleKvFormat;

impl KVFormat for ExampleKvFormat {
    fn name(&self) -> &str {
        "example_kv_format"
    }

    fn layout(&self) -> KVLayoutDesc {
        KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        }
    }
}

// Registration (dual-wiring) — static: linkme `KV_FORMATS`. Dynamic (`--features
// plugin-cdylib`): the `register_kv_format_v1` C-ABI export. One line wires both.
technique_api::register_kv_format!("example_kv_format", || Box::new(ExampleKvFormat));
// GATE-C v2: emit the `.so` entry (plugin-cdylib gate). The engine does not force-link it →
// a vehicle for the dynamic-registration success path (register_dynamic_plugins → make_format fallback).
technique_api::export_plugin!();

#[cfg(test)]
mod tests {
    use super::*;
    use technique_api::find_kv_format;

    #[test]
    fn registers_into_kv_formats() {
        let reg = find_kv_format("example_kv_format")
            .expect("example format must be registered in KV_FORMATS");
        assert_eq!(reg.name, "example_kv_format");
        let fmt = (reg.make)();
        assert_eq!(fmt.name(), "example_kv_format");
        let l = fmt.layout();
        assert_eq!(l.block_elems, 32);
        assert_eq!(l.bits, 4);
        assert_eq!(l.scale_layout, ScaleLayout::PerBlockF16);
        assert_eq!(l.packing, Packing::Nibble);
    }
}
