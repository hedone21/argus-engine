# Writing an Argus plugin

**English** | [한국어](plugins.ko.md)

A guide for adding a **KV-cache technique** to Argus without forking the engine: a new
eviction policy, a new storage format, a query-aware read strategy. You write one trait
impl in a small crate of your own, it registers itself, and the engine picks it up by
name. This is the same `technique-api` surface the built-in techniques (`q4_0`, `quest`,
`caote`, …) are built on.

> **New here?** Do the [Quickstart](#quickstart-your-first-plugin) (~10 minutes, runs on
> the shipped `argus-cli`), then stop. Everything after it is reference and how-to you can
> come back to one section at a time.

By the end of the Quickstart you will have a custom KV-cache format compiled into its own
crate, self-registered via [`linkme`](https://docs.rs/linkme), loaded into the real
inference binary with `--load-plugin`, and selected at runtime with `--kv-format my_format`,
with no rebuild of the engine.

---

## Contents

- [The mental model in 60 seconds](#the-mental-model-in-60-seconds)
- [Before you start](#before-you-start)
- [Quickstart: your first plugin](#quickstart-your-first-plugin)
- [Anatomy of a plugin](#anatomy-of-a-plugin)
- [The three extension axes](#the-three-extension-axes)
- [How-to: add an eviction / merge stage](#how-to-add-an-eviction--merge-stage)
- [How-to: add a read stage](#how-to-add-a-read-stage)
- [How-to: add a backend capability (KIVI)](#how-to-add-a-backend-capability-kivi)
- [Lifecycle & performance contract](#lifecycle--performance-contract)
- [Static vs dynamic: choosing a load path](#static-vs-dynamic-choosing-a-load-path)
- [The dynamic `.so` path in depth](#the-dynamic-so-path-in-depth)
- [The static (force-link) path](#the-static-force-link-path)
- [Troubleshooting](#troubleshooting)
- [Packaging & publishing checklist](#packaging--publishing-checklist)
- [Where to go next](#where-to-go-next)

---

## The mental model in 60 seconds

The whole system rests on one rule: **the plugin decides, the engine executes.** Your code
never touches a KV buffer or a GPU command queue directly. It reads a read-only view of
the cache and returns a *plan* (or a *descriptor*); the engine owns all mutation. That keeps
a technique crate decoupled from engine internals, and lets the same code link statically or
load as a `.so`.

Argus splits cache management into orthogonal axes (full vocabulary in
[`CONTEXT.md`](../CONTEXT.md)). A plugin extends exactly one axis:

| Axis | You implement | You return / describe |
|------|---------------|------------------------|
| **stage** | `KVCacheStage` (evict/merge) or `KVReadStage` (read) | which tokens to keep + how to merge, or which to read for attention |
| **format** | `KVFormat` | the byte layout / precision your cache is stored in (q4_0, KIVI) |
| **backend-capability** | `KiviAttentionBackend` | a specialized fused GPU kernel |

Each axis is independent: adding a member to one touches zero code on the others, and zero
code in the engine core.

> Why no central list to edit? Each technique submits itself to a global registry slice at
> link time via `linkme`. The engine reads the slice; there is no `match` arm to add. The
> deeper "why three axes" model lives in [`CONTEXT.md`](../CONTEXT.md); you do not need it to
> finish the Quickstart.

---

## Before you start

You need:

- A stable Rust toolchain and the `argus-engine` workspace cloned.
- One successful baseline build so you know your environment is sane:
  ```bash
  cargo build --release
  ```
- A model to run inference against (the CLI defaults to `models/llama3.2-1b`). Any model
  the engine already loads works; the plugin you write is model-agnostic. The repo does not
  ship a model. See the [README](../README.md) for how to obtain or convert one. Steps 1–3
  of the Quickstart need no model (they prove registration), so you can do the whole
  build-and-verify loop before you have one; only Step 4 runs inference.

The Quickstart is CPU-only and host-runnable: no GPU, no device. (GPU/on-device specifics
are out of scope here.)

> **One safety note for later.** The dynamic `.so` path loads native code into the engine
> process with no sandbox: a panic or segfault in a plugin crashes the engine, and the
> ABI is not stable across `technique-api` versions. None of that touches the Quickstart
> (you write only safe Rust), but keep it in mind before shipping a `.so` to someone else.

---

## Quickstart: your first plugin

We will add a KV-cache **format**, the axis that runs end-to-end on the shipped `argus-cli`
and needs no engine rebuild. The starting point is a maintained template crate,
`example-kv-format`, which you copy and rename.

### Step 1 — copy the template

```bash
cp -r crates/techniques/example-kv-format crates/techniques/my-format
```

Open `crates/techniques/my-format/Cargo.toml` and change only the package name:

```toml
[package]
name = "my-format"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[lib]
# cdylib = the dlopen-able `.so`/`.dylib`; rlib = the static force-link path.
crate-type = ["cdylib", "rlib"]

[dependencies]
technique-api = { path = "../../technique-api" }
linkme = "0.3"

[features]
# Enables the C-ABI export used by the dynamic `.so`. OFF by default; turn it on only
# when producing a loadable library (keeping it off for static builds avoids #[no_mangle]
# symbol collisions).
plugin-cdylib = []
```

Replace the body of `crates/techniques/my-format/src/lib.rs` with this complete file. The
only thing that matters is that the name string is unique:

```rust
//! my-format — a custom KV-cache format (a q4_0-like descriptor).

use technique_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

/// A format is a *pure descriptor*: name + byte layout. No compute lives here —
/// the engine's generic reader dequantizes through this layout to f32.
struct MyFormat;

impl KVFormat for MyFormat {
    fn name(&self) -> &str {
        "my_format"
    }

    fn layout(&self) -> KVLayoutDesc {
        // q4_0-like: 32 elements per block, 4 bits each, one f16 scale per block,
        // nibble-packed. (See KVLayoutDesc for the field meanings.)
        KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        }
    }
}

// One line registers MyFormat on BOTH paths: the static `KV_FORMATS` linkme slice and
// (under `--features plugin-cdylib`) the dynamic C-ABI export.
technique_api::register_kv_format!("my_format", || Box::new(MyFormat));

// Emits this crate's `.so` entry symbols. Needed only for the dynamic path; a no-op for
// static builds.
technique_api::export_plugin!();

#[cfg(test)]
mod tests {
    use technique_api::find_kv_format;

    #[test]
    fn registers_into_kv_formats() {
        let reg = find_kv_format("my_format").expect("my_format must be registered");
        assert_eq!(reg.name, "my_format");
    }
}
```

No workspace edit is needed: the root `Cargo.toml` globs `crates/techniques/*`, so your
copied folder is already a member.

### Step 2 — build it as a loadable library

```bash
cargo build --release -p my-format --features plugin-cdylib
```

> Build the plugin in the same profile as the engine (we used `--release` for both) so
> their ABIs match when the engine `dlopen`s it.

**Checkpoint.** The shared library is now under `target/release/`:

- Linux / Android: `target/release/libmy_format.so`
- macOS host: `target/release/libmy_format.dylib`

(The crate name's hyphen becomes an underscore in the library file name.) To print the
exact path programmatically, a variant of what the gate tests do (which build debug and
match `.so`):

```bash
cargo build --release -p my-format --features plugin-cdylib --message-format=json \
  | rg compiler-artifact | rg -o '"[^"]*libmy_format[^"]*\.(so|dylib)"'
```

### Step 3 — prove registration fired (before running inference)

`linkme` registration is invisible: a plugin that fails to register produces no compile
error, just a name the engine can't find. The template ships a unit test that proves the
slice was populated. Run it:

```bash
cargo test -p my-format
```

**Checkpoint.** `registers_into_kv_formats` passes. Your `impl KVFormat` is now discoverable
by `find_kv_format("my_format")`.

### Step 4 — run it in the real engine

```bash
cargo run --release -p argus-engine --bin argus-cli -- \
  -m models/llama3.2-1b \
  -p "Hello, world! I am a" \
  -n 20 \
  --load-plugin target/release/libmy_format.dylib \
  --kv-format my_format
```

(On macOS the built library is `libmy_format.dylib`; on Linux/Android it is
`libmy_format.so`. Use whichever path [Step 2](#step-2--build-it-as-a-loadable-library)
produced.)

**Checkpoint.** Decode runs to completion and prints the usual `Decode: X ms/tok` line.
That means the engine `dlopen`ed your `.so` and resolved `my_format` from its dynamic
registry. (Because this descriptor is bit-equivalent to the built-in `q4_0`, the engine then
routes storage to its typed `q4_0` fast path; a descriptor with no built-in equivalent (say
a different `block_elems`) instead drives the generic opaque-descriptor floor.)

If instead you see:

```
Unknown --kv-format 'my_format' (not found in either static KV_FORMATS or dynamic
registration — check --load-plugin)
```

…the name in `--kv-format` doesn't match a registered format, or `--load-plugin` points at
the wrong file. (See [Troubleshooting](#troubleshooting).)

You just added a KV-cache format the engine loads with a flag, no engine rebuild.

---

## Anatomy of a plugin

Every plugin is the same two-part shape: implement a trait, then register it in one line.

```rust
// 1. DECLARE behaviour by implementing the axis trait.
impl KVFormat for MyFormat {
    fn name(&self) -> &str { "my_format" }
    fn layout(&self) -> KVLayoutDesc { /* ... */ }
}

// 2. REGISTER. This single line is the entire wiring — there is no central file to edit.
technique_api::register_kv_format!("my_format", || Box::new(MyFormat));
technique_api::export_plugin!();   // only needed for the dynamic `.so` path
```

What that registration line actually does:

- **Static path (always).** It contributes a `KVFormatReg { name, make }` entry to the
  global `KV_FORMATS` `#[distributed_slice]`. When your crate is linked into a host binary,
  the linker gathers every such entry into one array; the engine looks yours up with
  `find_kv_format("my_format")`. No `match` arm, no registry file.
- **Dynamic path (under `--features plugin-cdylib`).** The same macro also emits an
  `unsafe extern "C"` vtable behind the feature gate. `export_plugin!()` emits the three
  per-`.so` entry symbols (`register_kv_stages_v2` / `register_kv_formats_v2` /
  `register_backend_caps_v2`) the host calls after `dlopen`. For a format-only crate, only
  the format one carries entries. You write only safe Rust; the `unsafe` C-ABI marshalling
  lives entirely inside the macro.

> **Factory arity differs by axis.** A format factory takes no arguments
> (`|| Box::new(MyFormat)`); a stage or read factory takes a params struct
> (`|p: StageParams| …` / `|p: ReadStageParams| …`).

So one line wires both ways, and which way is active is decided by how you build the
crate (plain `rlib` link vs. `--features plugin-cdylib` `.so`). That duality is the whole
[static-vs-dynamic](#static-vs-dynamic-choosing-a-load-path) story.

A note specific to **format**: `KVFormat` is a *pure descriptor* (just `name` + `layout`).
It carries no kernel. Novel precisions ride the engine's generic `dequant → f32` floor
using the `KVLayoutDesc` you return; a hand-written paired kernel is a separate, backend-
owned concern (see [`CONTEXT.md`](../CONTEXT.md) on "paired kernels"). The other axes
(`KVCacheStage`, `KVReadStage`) instead return a plan computed from a read-only
[`StageCtx`](#how-to-add-an-eviction--merge-stage).

---

## The three extension axes

"Selector kind" and "Runs on" matter as much as the trait: they decide whether the plugin
you write can actually be invoked from the binary you have.

| Axis | Trait | Returns | Register with | CLI selector | Selector kind | Dynamic `.so`? | Runs on | Maturity |
|------|-------|---------|---------------|--------------|---------------|----------------|---------|-----------|
| **format** | `KVFormat` | layout descriptor | `register_kv_format!` | `--kv-format <name>` | free string | **yes** | `argus-cli`, `argus-bench` | production |
| **stage** (read) | `KVReadStage` | `Option<KVReadPlan>` | raw `#[distributed_slice(KV_READ_STAGES)]` | `--read-stage <name>` | free string | no (static only) | `argus-cli` | production (`quest`) |
| **stage** (eviction/merge) | `KVCacheStage` | `Option<KVCachePlan>` | `register_kv_stage!` | `eviction plugin --name <n>` | free string | **yes** | `argus-bench`, `argus-eval` | production |
| **backend-cap** (KIVI attn) | `KiviAttentionBackend` | kernel result code | `register_kivi_attention_plugin!` | `--backend-cap <name>` | free string | **yes** (OpenCL) | `argus-bench`, `argus-eval` | production (OpenCL) |

All three axes select a registered technique by name with zero engine edits; they only
differ in *which binary* runs them and the exact flag spelling:

1. `--kv-format` and `--read-stage` are free string flags on the shipped `argus-cli`. This
   is the cleanest "add a folder, select by name" story, and why the Quickstart uses format.

2. Stage and backend-capability are selected by name on `argus-bench` / `argus-eval`. A
   plugin eviction stage uses `eviction plugin --name <n>` (the `eviction` subcommand's
   free-string escape hatch); a backend capability uses `--backend-cap <name>` (OpenCL). Both
   resolve through the same static→dynamic registry lookup as format. `argus-cli` doesn't run
   eviction or KIVI in v0, so those two axes live on the bench/eval binaries. The resilience
   manager then drives a CLI-selected stage (evict timing/ratio); switching the technique
   *by name at runtime* over the manager IPC is not yet supported (it needs an `argus-shared`
   protocol addition). See the [stage how-to](#how-to-add-an-eviction--merge-stage).

> Beyond these three, a further **weight** axis (`WeightStage` / `WEIGHT_STAGES`, for runtime
> weight swap) exists. Its trait surface is unit-tested, and can be registered today via the raw
> `#[distributed_slice(WEIGHT_STAGES)]` path (there is no convenience macro), but it has
> no CLI flag and no shipped consumer yet, so treat it as experimental and not part of
> the usable plugin surface.

---

## How-to: add an eviction / merge stage

The **stage** axis is the richest API and the best way to understand the "plugin decides,
engine executes" contract. A stage reads a read-only view of the cache and returns a plan
of which tokens to keep and how to merge. It never mutates a buffer.

Template: [`crates/techniques/example-keep-recent`](../crates/techniques/example-keep-recent)
(keep the most recent `target_len` tokens, the prefix-0 case of sliding-window).

```rust
use technique_api::{KVCachePlan, KVCacheStage, KeepSpec, StageCtx, StageParams};

struct KeepRecent;

impl KVCacheStage for KeepRecent {
    fn name(&self) -> &str {
        "example_keep_recent"
    }

    /// Called when the engine needs an eviction decision. Return `None` for a no-op.
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let current = ctx.current_pos();   // valid tokens right now
        let target = ctx.target_len();     // budget the engine resolved for you
        if current <= target {
            return None;                   // nothing to shrink
        }
        // Keep the most recent `target` tokens (positions ascending).
        let keep: Vec<usize> = (current - target..current).collect();
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep),
            merges: Vec::new(),            // no weighted merges
        })
    }
}

technique_api::register_kv_stage!(
    "example_keep_recent",
    |_params: StageParams| Box::new(KeepRecent)
);
technique_api::export_plugin!();
```

### What you can read — `StageCtx`

`StageCtx` is the read-only window onto the cache. Because `plan` takes it as `&dyn
StageCtx`, every accessor is object-safe (outputs go through `&mut [f32]` out-params, not
slice returns). The accessors:

| Accessor | Meaning |
|----------|---------|
| `current_pos()` | number of valid tokens |
| `target_len()` | the budget (token count) the engine resolved from the ratio/limit |
| `layer_idx()` | which layer this call is for (per-layer decisions) |
| `n_kv_heads()`, `head_dim()` | cache geometry |
| `importance()` → `Option<&[f32]>` | flat per-token importance (`Some` for H2O-style scoring; `None` for score-free policies) |
| `tensor(TensorKind)` → `Option<&dyn TensorHandle>` | raw `Key` / `Value` / `AttnWeights` / `Scores` / `QueryStats`, read row-by-row |

The `tensor()` accessor is the single mechanism for everything heavy; `dequant_k`,
`dequant_v`, `head_score`, `attn_weight` are default sugar over it. A stage computes its
own metric from these reads. The built-in `caote`
([`crates/techniques/caote`](../crates/techniques/caote)) reads `Value` and computes a
value-aware criticality score the engine never had to provide.

### What you return — `KVCachePlan`

- `keep: KeepSpec` — `LayerWide(Vec<usize>)` (same tokens across all heads, ascending) or
  `PerHead(Vec<Vec<usize>>)` (one ascending list per KV head, e.g. H2O+).
- `merges: Vec<WeightedMerge>` — optional weighted merges (D2O folds evicted tokens into a
  retained slot with magnitude-preserving weights). Empty for evict-only policies.

The engine takes this plan and performs the actual buffer rewrite via `compact`. You hold
any cross-call state (EMA accumulators, page metadata) on `&self` with interior
mutability (a `Mutex`), since `plan` takes `&self`. See how the built-in `quest`
(`Mutex<QuestState>`) and the in-engine `d2o` stage (`engine/src/kv/d2o_handler.rs`, not a
technique crate) hold theirs. (`caote`, by contrast, is stateless: it recomputes from `ctx`
on every call.)

### Selecting your stage

A plugin stage is selected by name, on the eviction-capable binaries:

```bash
# build the stage .so (see the dynamic path), then load + select it by name:
argus-bench ... --load-plugin target/release/libmy_stage.so eviction plugin --name my_stage
# or, if you force-linked it statically, just name it:
argus-eval  ... eviction plugin --name my_stage
```

`eviction plugin --name <name>` is the `eviction` subcommand's free-string escape hatch,
the stage-axis analogue of `--kv-format`. The name resolves through `make_stage` (static
`KV_CACHE_STAGES` first, then the `--load-plugin` dynamic registry), so any registered stage
is selectable with no engine edit. (The built-in policies keep their own typed
subcommands: `eviction sliding`, `eviction h2o`, ….)

Two scoping notes:

- **Bench/eval, not `argus-cli`.** Eviction needs a cache manager, which only `argus-bench`
  and `argus-eval` build; `argus-cli` (the single-prompt happy path) rejects eviction in v0.
- **Runtime by-name switch doesn't exist yet.** The resilience manager
  drives a CLI-selected stage (evict timing/ratio); selecting a *different* technique by name
  at runtime over the manager IPC would need a new `argus-shared` command (out of scope today).

> Some `technique-api` doc-comments still call the selector `--eviction-policy <name>`, a
> legacy name; the real CLI form is `eviction plugin --name <name>` (or a built-in
> `eviction <policy>` subcommand).

---

## How-to: add a read stage

A **read stage** (the read kind of the stage axis) decides *what to read* for attention, a
query-aware subset of tokens or pages, without changing what's stored. `None` means "full
read" (exact, the default), so a read stage is an opt-in approximation, never a correctness
risk.

Reference: the built-in [`crates/techniques/quest`](../crates/techniques/quest) (Quest,
ICML'24, keeps per-page K min/max and reads only the top-k pages by an upper-bound dot
product).

```rust
use linkme::distributed_slice;
use technique_api::{
    KV_READ_STAGES, KVReadPlan, KVReadStage, KVReadStageReg, ReadGranularity,
    ReadStageParams, StageCtx,
};

struct MyRead { /* page metadata held here, behind a Mutex */ }

impl KVReadStage for MyRead {
    fn name(&self) -> &str { "my_read" }

    /// Fired once per layer, just before attention. `None` = full read for this layer.
    fn read_plan(&self, ctx: &dyn StageCtx) -> Option<KVReadPlan> {
        // ... pick positions/pages from ctx.tensor(Key)/ctx.current_pos() ...
        Some(KVReadPlan {
            granularity: ReadGranularity::Page { page_size: 16 },
            select: vec![/* ascending page indices */],
        })
    }
}

// A read stage has no convenience macro: submit the registration entry directly.
#[distributed_slice(KV_READ_STAGES)]
static MY_READ: KVReadStageReg = KVReadStageReg {
    name: "my_read",
    make: |p: ReadStageParams| Box::new(MyRead { /* ... */ }),
};
```

Two things to know:

- **Selected with a free string flag**, `--read-stage my_read`, and it reaches decode on
  the shipped `argus-cli`. (Default unset = full read = byte-identical to no plugin.)
- **Static only.** There is no dynamic `.so` entry symbol for read stages, so a read
  stage must be force-linked (see [the static path](#the-static-force-link-path)); it
  cannot be added via `--load-plugin`. `quest` is wired this way as a non-optional
  dependency.

> A read plan only takes effect when the active format supports selective reads (the
> default `StandardFormat` does). Pairing `--read-stage` with an opaque custom `--kv-format`
> falls back to a full read with a one-time stderr warning: correct, just not accelerated.

---

## How-to: add a backend capability (KIVI)

> **Advanced / experimental.** This axis is GPU-specialized and not reachable from the
> shipped `argus-cli`.

The **backend-capability** axis layers a specialized fused kernel onto a backend. The
in-tree instance is KIVI fused dequant+attention. You implement `KiviAttentionBackend`
(building your OpenCL kernels once from a borrowed GPU context) and register with
`register_kivi_attention_plugin!`. Reference:
[`crates/techniques/example-backend-cap`](../crates/techniques/example-backend-cap), which
exercises the full ABI round-trip with no real GPU math.

Select a capability by name with `--backend-cap <name>`:

```bash
argus-bench ... --kv-mode kivi --load-plugin target/release/libmy_kivi.so --backend-cap my_kivi
```

`--backend-cap <name>` resolves a `KiviAttentionBackend` by registry name (static
`KIVI_ATTENTION_REGS` first, then the `--load-plugin` dynamic registry) and installs it in
place of the engine's built-in OpenCL implementation. Unset = the built-in. It is the
backend-capability axis's analogue of `--kv-format`.

Scoping notes:

- **OpenCL-only, bench/eval-only.** The capability is registered only on the OpenCL backend
  (the CPU/CUDA arms have none, so `--backend-cap` there warns and is ignored); `--kv-mode
  kivi` is rejected by `argus-cli` in v0.
- **Real kernels need a device.** A plugin `KiviAttentionBackend` runs actual GPU code, so
  on-host CI verifies only compilation + the synthetic gate test
  (`engine/tests/gate_c_backend_cap_dlopen.rs`); real correctness is on-device (Adreno/Mali).

---

## Lifecycle & performance contract

Knowing *when* your methods run is what separates a working plugin from one that quietly
destroys decode throughput.

| Method | When it runs | Hot path? |
|--------|--------------|-----------|
| factory `make(...)` | once, at session/cache construction | no, do heavy setup here |
| `KVFormat::layout()` | once, at cache construction | no, but the descriptor it returns is read on the hot path, so keep it a trivial POD return |
| `KVCacheStage::plan()` | when the engine triggers an eviction (budget/pressure) | warm, runs during decode |
| `KVReadStage::read_plan()` | once per layer, just before attention | **yes**, per layer, per token |

The performance contract for warm/hot methods:

- Do not allocate, lock contended state, or log on the hot path. Pre-size buffers and
  build kernels in the factory.
- Hold state on `&self` via interior mutability (`Mutex<...>`), as `plan`/`read_plan`
  take `&self`. Mirror `quest`'s `Mutex<QuestState>` page metadata (the in-engine `d2o`
  stage holds its EMA the same way).
- Cleanup is RAII. Implement `Drop` if you own resources; the engine drops your
  instance when the cache/session ends.
- Measure without `--profile`. Per repo policy, `--profile` adds ~54 ms/token of sync
  overhead and is rejected by the happy-path binaries. Read real throughput from the
  `Decode: X ms/tok` log line; use `--profile` only for *relative* per-op comparison.

---

## Static vs dynamic: choosing a load path

Every `register_*!` macro wires a technique both ways. You choose at build time which is
active.

| | **Static** (linkme force-link) | **Dynamic** (`cdylib` `.so` + `dlopen`) |
|--|-------------------------------|------------------------------------------|
| Add a technique by | recompiling the host with your crate linked | `--load-plugin foo.so` at startup, no host rebuild |
| Hot-swappable | no | yes |
| Runtime overhead | zero (direct call) | one `dlopen` + indirect C-ABI call |
| Type richness | full Rust types | flat C-ABI structs (handled inside the macros) |
| ABI stability | n/a (same compile) | fragile, must match `technique-api` version |
| Crash isolation | shares the process | shares the process (no sandbox) |
| Who ships it | the engine maintainer | anyone, as a standalone `.so` |
| Axes supported | all three | format, stage, backend-cap (read is static-only) |

How a compile-time registry picks up a runtime plugin: the two paths register at *different
times*. Static elements are gathered into the linker section at link time; a `.so`'s
capabilities are registered when `dlopen` runs. At lookup time the engine checks the static
slice first, then falls back to the dynamic registry, so both resolve through one
source-agnostic `find_*` / `make_*` seam. One line (`register_*!` + `export_plugin!()`),
gated by the `plugin-cdylib` feature and `crate-type = ["cdylib", "rlib"]`, serves both.

Rule of thumb: develop and ship built-ins statically; use the dynamic path when you want
to distribute a technique without rebuilding the engine (today that's cleanest for the
format axis, which is selectable by a free string flag).

---

## The dynamic `.so` path in depth

Building and loading a `.so`, end to end:

```bash
# Build (same profile as the engine):
cargo build --release -p my-format --features plugin-cdylib
# -> target/release/libmy_format.{so|dylib}

# Load at startup (repeatable; each .so is dlopen'd once and routed to every axis it
# exports) and select:
argus-cli ... --load-plugin target/release/libmy_format.so --kv-format my_format
```

What happens under the hood:

- `export_plugin!()` always emits the same three `#[no_mangle] extern "C"` entry symbols per
  `.so`, one per dynamic axis: `register_kv_stages_v2`, `register_kv_formats_v2`,
  `register_backend_caps_v2`. (There is no read-stage symbol; read stays static.) An
  axis you didn't use simply exports a count of 0; absence is not an error.
- `--load-plugin` `dlopen`s each path once and routes the handle to all three axis
  registrars. A `.so` can therefore bundle multiple axes: see
  [`example-bundle`](../crates/techniques/example-bundle) (a stage + a format in one `.so`)
  and [`example-multi-format`](../crates/techniques/example-multi-format) (two formats).
- **ABI handshake.** Each envelope carries an `abi_version`; the host rejects the `.so`
  fail-fast on mismatch (`KV_STAGE`/`KV_FORMAT_ABI_VERSION = 2`, `BACKEND_CAP = 1`). Rebuild
  your plugin against the engine's `technique-api` version.
- **Name collisions.** A dynamic name that collides with a built-in (or another loaded
  plugin) is rejected; the built-in wins. Registration of one `.so` is 2-pass atomic, so a
  partial/conflicting envelope registers nothing (see
  [`example-rollback`](../crates/techniques/example-rollback)).
- **Forgot `export_plugin!()`?** The `.so` exports no entry symbols, contributes 0
  capabilities, and the loader bails (see
  [`example-no-export`](../crates/techniques/example-no-export)).

**Safety.** A `.so` runs in-process with no sandbox: a panic or segfault crashes the engine.
Keep the `plugin-cdylib` feature off for static builds, or the `#[no_mangle]` C-ABI
symbols collide with the force-linked copy.

Platform note: `crate-type = ["cdylib"]` produces `lib<name>.so` on Linux/Android and
`lib<name>.dylib` on a macOS host.

---

## The static (force-link) path

This is how the always-linked built-ins (`quest`, `synth-q4-format`) are wired. It is also the
route you use for a read stage (which has no dynamic path) or any technique you want
compiled into the engine. (The `caote` stage is wired the same way but behind a cargo
feature; see [How-to: stage](#how-to-add-an-eviction--merge-stage).)

1. Add your crate as a path dependency in `engine/Cargo.toml`:
   ```toml
   my-format = { path = "../crates/techniques/my-format" }
   ```
2. **Force-link it.** Rust drops an unreferenced rlib (and with it your `#[distributed_slice]`
   registration), so add a reference line at the axis's designated force-link site, e.g.
   `use my_format as _;` (formats: `engine/src/format/builtin_kv_formats.rs`; read stages:
   `engine/src/kv/read/read_stage_registry.rs`).
3. The engine's startup self-test asserts your name is present (guarding against fat-LTO /
   `--gc-sections` silently dropping the registration). A missing name fails fast at startup
   rather than silently disabling the technique.

A static technique needs no `--features plugin-cdylib` and no `--load-plugin`; it's resolved
directly from the linkme slice by `find_kv_format` / `find_read_stage` / `find_stage`.

---

## Troubleshooting

These are the system's *signature* failure modes, the ones with no obvious compiler error.

| Symptom | Cause | Fix |
|---------|-------|-----|
| Plugin compiles but the engine can't find the name | The crate isn't in the host's link graph, or its registration was dead-code-eliminated | For the static path, add the path dep **and** a `use my_crate as _;` force-link line; for the dynamic path, pass `--load-plugin <.so>` |
| `Unknown --kv-format '…' (… check --load-plugin)` | Name typo, or `--load-plugin` points at the wrong/old `.so` | Confirm the `name()` string equals the flag value; rebuild and point at the fresh `target/<profile>/lib*.so` |
| `.so` loads but `0 capabilities registered (… export_plugin! missing?)` | Forgot `technique_api::export_plugin!()` | Add it once per `.so` ([`example-no-export`](../crates/techniques/example-no-export)) |
| `.so` rejected: `abi_version … != expected …` | Plugin built against a different `technique-api` version | Rebuild the plugin against the engine's version (ABI is not stable across versions) |
| Duplicate-name rejection on load | The name collides with a built-in or another plugin | Rename your technique ([`example-rollback`](../crates/techniques/example-rollback) shows the atomic-rollback behavior) |
| Linker `#[no_mangle]` symbol collision | A crate is force-linked **and** built with `--features plugin-cdylib` | Keep `plugin-cdylib` off for static builds; turn it on only when producing a `.so` |
| Custom eviction stage won't select | Using `argus-cli` (no eviction in v0), or a name typo | Select on `argus-bench`/`argus-eval` with `eviction plugin --name <name>`; confirm the name matches `name()`; see [Selecting your stage](#how-to-add-an-eviction--merge-stage) |
| Decode got slower after adding a stage/read plugin | Allocating/locking/logging on the hot path | Move setup into the factory; hold state on `&self`; measure with `Decode: X ms/tok`, not `--profile` |

---

## Packaging & publishing checklist

Turning "I implemented the trait" into "I shipped a crate someone can load":

- [ ] `Cargo.toml`: `crate-type = ["cdylib", "rlib"]` and a `plugin-cdylib = []` feature
      (off by default).
- [ ] Dependencies are exactly `technique-api` (path or version) + `linkme = "0.3"`, with no
      engine dependency.
- [ ] A unit test that asserts registration fired (`find_kv_format(...).is_some()` etc.).
- [ ] State the `technique-api` version your `.so` is built against; the dynamic ABI is not
      stable across versions.
- [ ] License: contributions to this repo are dual `MIT OR Apache-2.0`; match it for
      in-tree crates.
- [ ] Pick a unique, descriptive `name()` string (it's the CLI selector and the registry
      key).

---

## Where to go next

- **Reference:** the full trait/macro surface with doc comments:
  [`crates/technique-api/src/lib.rs`](../crates/technique-api/src/lib.rs) (run `cargo doc
  -p technique-api --open`) and the crate's own
  [`README.md`](../crates/technique-api/README.md).
- **Concepts / vocabulary:** [`CONTEXT.md`](../CONTEXT.md): the stage ⊥ format ⊥ hardware
  model, why a *format* is never a *layer*, and why composite operations decompose into
  single-axis primitives.
- **Templates, mapped to their axis:**

  | Use as a template for… | Crate |
  |------------------------|-------|
  | a **stage** (eviction/merge) | [`example-keep-recent`](../crates/techniques/example-keep-recent) |
  | a **format** | [`example-kv-format`](../crates/techniques/example-kv-format) |
  | a metric-computing real stage | [`caote`](../crates/techniques/caote) |
  | a real read stage | [`quest`](../crates/techniques/quest) |
  | multiple axes in one `.so` | [`example-bundle`](../crates/techniques/example-bundle) |
  | multiple formats in one `.so` | [`example-multi-format`](../crates/techniques/example-multi-format) |
  | a backend capability (KIVI) | [`example-backend-cap`](../crates/techniques/example-backend-cap) |
  | the failure paths (no-export, name collision) | [`example-no-export`](../crates/techniques/example-no-export), [`example-rollback`](../crates/techniques/example-rollback) |
