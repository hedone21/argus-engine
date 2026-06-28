# argus-extension-api

The additive extension surface of argus-engine. A new KV-cache technique is a separate
crate under [`crates/techniques/`](..) that depends only on this crate and self-registers
via [`linkme`](https://docs.rs/linkme): adding a folder requires zero edits to the engine
core.

Three orthogonal extension axes (see [`../../CONTEXT.md`](../../CONTEXT.md) for the
vocabulary: **stage** ⊥ **format** ⊥ **hardware**). This crate exposes the registration
surface for all three:

| Axis | Trait | Register with | Engine looks it up via |
|------|-------|---------------|------------------------|
| stage (resident-token adjustment) | [`KVMutationStage`] (evict/merge) or [`KVReadStage`] (query-aware read) | `register_kv_mutation_stage!` macro, or `#[distributed_slice(KV_READ_STAGES)]` (read) | `find_mutation_stage(name)` / `find_read_stage(name)` |
| format (storage representation) | [`KVFormat`] | `register_kv_format!` macro | `find_kv_format(name)` |
| backend-capability (fused GPU kernel) | [`QuantAttnBackend`] | `register_quant_attn_plugin!` macro | (dynamic backend registry) |

A technique never references engine types (`KVCache`/`Backend`); it reads cache state through
the abstractions this crate defines (e.g. [`StageCtx`]) and stages mutations imperatively on a
transactional [`CacheHandle`]. The engine owns the commit.

> **New here?** This README is the terse registration reference. For a hands-on onboarding
> guide that copies a template crate and selects it at runtime, see
> [`docs/plugins.md`](../../docs/plugins.md) ([한국어](../../docs/plugins.ko.md)).

## Add a KV-cache stage

1. Create `crates/techniques/<name>/` with a `Cargo.toml` depending on `argus-extension-api` and
   `linkme`.
2. Implement [`KVMutationStage`] and register it in one line:

```rust
use argus_extension_api::{
    CacheHandle, CacheOpError, KVMutationStage, MutationPhase, StageCtx,
};

struct KeepRecent;

impl KVMutationStage for KeepRecent {
    fn name(&self) -> &str { "example_keep_recent" }

    fn on_phase(&self, ctx: &dyn StageCtx, cache: &mut dyn CacheHandle)
        -> Result<(), CacheOpError>
    {
        let (current, target) = (ctx.current_pos(), ctx.target_len());
        if current <= target { return Ok(()); } // no-op
        cache.keep(&(current - target..current).collect::<Vec<_>>())
    }
}

argus_extension_api::register_kv_mutation_stage!(
    "example_keep_recent", |_p| Box::new(KeepRecent), MutationPhase::KvMutate);
```

Working template: [`crates/techniques/example-keep-recent`](../techniques/example-keep-recent).

> Knobs beyond `StageParams`'s five common fields (e.g. d2o's `merge_axis`/`protected_layers`)
> ride an opaque `key=value` blob: register a direct `MutationStageReg` literal with a
> `make(StageParams, StageArgs)` factory and parse the blob in the plugin (the engine never
> names your config type). See [`docs/plugins.md`](../../docs/plugins.md) → *Passing
> technique-private params*; `crates/techniques/d2o` is the worked example.

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

A read stage is the read kind of the stage axis. It has no convenience macro: submit a
[`KVReadStageReg`] to the [`KV_READ_STAGES`] slice directly.

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

- **Static** (default, and the ONLY path for KV mutation stages): the technique links into the
  engine and is discovered through the `KV_MUTATION_STAGES` / `KV_FORMATS` / `KV_READ_STAGES`
  slices at construction. A one-line path dependency on your crate force-links it.
- **Dynamic** (`--features plugin-cdylib`, **format + backend-capability axes only**): the crate
  builds as a `cdylib`; `export_plugin!()` emits the `.so` format/backend entry points so the host
  can `dlopen` it with no recompile, e.g. `argus-bench --load-plugin <plugin>.so --kv-format <name>`
  (a backend capability uses `--backend-cap <name>`). KV mutation stages have no `.so` C-ABI — a
  `CacheHandle` is a `&mut dyn` transaction engine, not a flat fn-ptr table — so they register
  statically only and are selected with `eviction plugin --name <name>`.

Keep `plugin-cdylib` off for static builds so the `#[no_mangle]` C-ABI symbols won't
collide with a force-linked copy.

See the `example-*` crates for bundles (a static KV stage + a dynamic format in one crate),
multi-format plugins, and rollback/error-path vehicles.
