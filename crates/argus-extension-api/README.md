# argus-extension-api

The **additive extension surface** of argus-engine. A new KV-cache technique is a separate
crate under [`crates/techniques/`](..) that depends only on this crate and self-registers
via [`linkme`](https://docs.rs/linkme) — **adding a folder requires zero edits to the engine
core**.

There are three orthogonal extension axes (see [`../../CONTEXT.md`](../../CONTEXT.md) for the
full vocabulary — **stage** ⊥ **format** ⊥ **hardware**). This crate exposes the registration
surface for three of them:

| Axis | Trait | Register with | Engine looks it up via |
|------|-------|---------------|------------------------|
| stage (resident-token adjustment) | [`KVCacheStage`] | `register_kv_stage!` macro | `find_stage(name)` |
| format (storage representation) | [`KVFormat`] | `register_kv_format!` macro | `find_kv_format(name)` |
| read (query-aware selective read) | [`KVReadStage`] | `#[distributed_slice(KV_READ_STAGES)]` (no macro) | `find_read_stage(name)` |

A technique never references engine types (`KVCache`/`Backend`); it reads cache state through
the abstractions this crate defines (e.g. [`StageCtx`]) and returns a *plan* — the engine owns
all buffer mutation.

> **New here?** This README is the terse registration reference. For a step-by-step
> onboarding guide — copy a template crate, build it, load it, select it at runtime, and
> ship it — see [`docs/plugins.md`](../../docs/plugins.md)
> ([한국어](../../docs/plugins.ko.md)).

## Add a KV-cache stage

1. Create `crates/techniques/<name>/` with a `Cargo.toml` depending on `argus-extension-api` and
   `linkme`.
2. Implement [`KVCacheStage`] and register it in one line:

```rust
use argus_extension_api::{KVCachePlan, KVCacheStage, KeepSpec, StageCtx, StageParams};

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

argus_extension_api::register_kv_stage!("example_keep_recent", |_p: StageParams| Box::new(KeepRecent));
argus_extension_api::export_plugin!(); // only needed for the dynamic `.so` path (see below)
```

Working template: [`crates/techniques/example-keep-recent`](../techniques/example-keep-recent).

## Add a KV-cache format

Implement [`KVFormat`] and register with `register_kv_format!`:

```rust
use argus_extension_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

struct ExampleKvFormat;

impl KVFormat for ExampleKvFormat {
    fn name(&self) -> &str { "example_kv_format" }
    fn layout(&self) -> KVLayoutDesc {
        KVLayoutDesc { block_elems: 32, bits: 4,
            scale_layout: ScaleLayout::PerBlockF16, packing: Packing::Nibble }
    }
}

argus_extension_api::register_kv_format!("example_kv_format", || Box::new(ExampleKvFormat));
argus_extension_api::export_plugin!();
```

Working template: [`crates/techniques/example-kv-format`](../techniques/example-kv-format).

## Add a read stage

The read axis has **no convenience macro** — submit a [`KVReadStageReg`] to the
[`KV_READ_STAGES`] slice directly:

```rust
use linkme::distributed_slice;
use argus_extension_api::{KV_READ_STAGES, KVReadStageReg};

#[distributed_slice(KV_READ_STAGES)]
static MY_READ_STAGE: KVReadStageReg = KVReadStageReg {
    name: "my_read_stage",
    // ...remaining fields (see KVReadStageReg)...
};
```

Working reference: the `quest` builtin in
[`crates/techniques/quest`](../techniques/quest).

## Static vs dynamic (`.so`)

Each `register_*!` macro wires the technique **both** ways:

- **Static** (default): the technique is linked into the engine and discovered through the
  `KV_CACHE_STAGES` / `KV_FORMATS` / `KV_READ_STAGES` slices at construction. Add a one-line
  path dependency on your crate to force-link it.
- **Dynamic** (`--features plugin-cdylib`): the crate builds as a `cdylib`; `export_plugin!()`
  emits the `.so` entry point so the host can `dlopen` it with no recompile, e.g.
  `argus-bench --load-plugin <plugin>.so eviction plugin --name <name>` (a format uses
  `--kv-format <name>`; a backend capability uses `--backend-cap <name>`).

Keep `plugin-cdylib` **off** for static builds so the `#[no_mangle]` C-ABI symbols do not
collide with a force-linked copy.

See the other `example-*` crates for bundles (stage + format in one `.so`), multi-format
plugins, and rollback/error-path vehicles.
