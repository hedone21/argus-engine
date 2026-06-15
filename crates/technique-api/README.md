# technique-api

The additive extension surface of argus-engine. A new KV-cache technique is a separate
crate under [`crates/techniques/`](..) that depends only on this crate and self-registers
via [`linkme`](https://docs.rs/linkme): adding a folder requires zero edits to the engine
core.

Three orthogonal extension axes (see [`../../CONTEXT.md`](../../CONTEXT.md) for the
vocabulary: **stage** ⊥ **format** ⊥ **hardware**). This crate exposes the registration
surface for all three:

| Axis | Trait | Register with | Engine looks it up via |
|------|-------|---------------|------------------------|
| stage (resident-token adjustment) | [`KVCacheStage`] (evict/merge) or [`KVReadStage`] (query-aware read) | `register_kv_stage!` macro, or `#[distributed_slice(KV_READ_STAGES)]` (read) | `find_stage(name)` / `find_read_stage(name)` |
| format (storage representation) | [`KVFormat`] | `register_kv_format!` macro | `find_kv_format(name)` |
| backend-capability (fused GPU kernel) | [`KiviAttentionBackend`] | `register_kivi_attention_plugin!` macro | (dynamic backend registry) |

A technique never references engine types (`KVCache`/`Backend`); it reads cache state through
the abstractions this crate defines (e.g. [`StageCtx`]) and returns a *plan*. The engine owns
all buffer mutation.

> **New here?** This README is the terse registration reference. For a hands-on onboarding
> guide that copies a template crate and selects it at runtime, see
> [`docs/plugins.md`](../../docs/plugins.md) ([한국어](../../docs/plugins.ko.md)).

## Add a KV-cache stage

1. Create `crates/techniques/<name>/` with a `Cargo.toml` depending on `technique-api` and
   `linkme`.
2. Implement [`KVCacheStage`] and register it in one line:

```rust
use technique_api::{KVCachePlan, KVCacheStage, KeepSpec, StageCtx, StageParams};

struct KeepRecent;

impl KVCacheStage for KeepRecent {
    fn name(&self) -> &str { "example_keep_recent" }

    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let (current, target) = (ctx.current_pos(), ctx.target_len());
        if current <= target { return None; } // no-op
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide((current - target..current).collect()),
            merges: Vec::new(),
        })
    }
}

technique_api::register_kv_stage!("example_keep_recent", |_p: StageParams| Box::new(KeepRecent));
technique_api::export_plugin!(); // only needed for the dynamic `.so` path (see below)
```

Working template: [`crates/techniques/example-keep-recent`](../techniques/example-keep-recent).

## Add a KV-cache format

Implement [`KVFormat`] and register with `register_kv_format!`:

```rust
use technique_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

struct ExampleKvFormat;

impl KVFormat for ExampleKvFormat {
    fn name(&self) -> &str { "example_kv_format" }
    fn layout(&self) -> KVLayoutDesc {
        KVLayoutDesc { block_elems: 32, bits: 4,
            scale_layout: ScaleLayout::PerBlockF16, packing: Packing::Nibble }
    }
}

technique_api::register_kv_format!("example_kv_format", || Box::new(ExampleKvFormat));
technique_api::export_plugin!();
```

Working template: [`crates/techniques/example-kv-format`](../techniques/example-kv-format).

## Add a read stage

A read stage is the read kind of the stage axis. It has no convenience macro: submit a
[`KVReadStageReg`] to the [`KV_READ_STAGES`] slice directly.

```rust
use linkme::distributed_slice;
use technique_api::{KV_READ_STAGES, KVReadStageReg};

#[distributed_slice(KV_READ_STAGES)]
static MY_READ_STAGE: KVReadStageReg = KVReadStageReg {
    name: "my_read_stage",
    // ...remaining fields (see KVReadStageReg)...
};
```

Working reference: the `quest` builtin in
[`crates/techniques/quest`](../techniques/quest).

## Static vs dynamic (`.so`)

Each `register_*!` macro wires the technique both ways:

- **Static** (default): the technique links into the engine and is discovered through the
  `KV_CACHE_STAGES` / `KV_FORMATS` / `KV_READ_STAGES` slices at construction. A one-line path
  dependency on your crate force-links it.
- **Dynamic** (`--features plugin-cdylib`): the crate builds as a `cdylib`; `export_plugin!()`
  emits the `.so` entry point so the host can `dlopen` it with no recompile, e.g.
  `argus-bench --load-plugin <plugin>.so eviction plugin --name <name>` (a format uses
  `--kv-format <name>`; a backend capability uses `--backend-cap <name>`).

Keep `plugin-cdylib` off for static builds so the `#[no_mangle]` C-ABI symbols won't
collide with a force-linked copy.

See the `example-*` crates for bundles (stage + format in one `.so`), multi-format
plugins, and rollback/error-path vehicles.
