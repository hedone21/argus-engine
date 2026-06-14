# Argus — KV/Weight Cache Management Context

**English** | [한국어](CONTEXT.ko.md)

A glossary for *how* KV and weight data are handled while the decode loop runs. It
covers three orthogonal axes — **stage** (adjusting memory-resident data) ⊥ **format**
(the precision/representation a computation runs in) ⊥ **hardware** (where a computation
runs). The three are independent by design, so the cost of combining members is a sum
(M+N+K), not a product (M×N×K).

## Language

### The three axes

**stage axis**:
The axis that adjusts *data resident in memory* (kvcache tokens, weight layers) while
decode runs — modify, delete, load/unload. It controls "what is loaded into memory, and
how much." Example members: H2O (delete tokens = evict), D2O (merge tokens = modify),
weight swap (load/unload a layer).

**format axis**:
The axis of *which data representation (precision)* each computation runs in. It controls
"in which format this computation is calculated." One member = one representation = a byte
layout + precision + the dedicated kernel that reads it (**paired kernel**). Examples: f16,
q4_0, q8_0, KIVI (Q2 packed + F32 residual). Precision and layout are one coordinate that
has never been separable (q4_0 *is* both a 4-bit block layout and a precision).

**hardware axis** (new):
The axis of *which physical compute unit* a computation executes on. It controls "where
this computation is calculated." A member = a backend (CPU NEON / Adreno OpenCL / Jetson
CUDA / NPU), and it carries the memory the data lives in. At runtime a [`Hardware`](#hardware--compute-location)
object (a read-only resolver) interprets the coordinate.

**orthogonality**:
The three axes are mutually independent. Adding a member to one axis does not touch the
others' code — adding a stage costs 0 kernels, adding a format costs a per-backend opt-in
kernel, adding a hardware costs a resolver entry + an opt-in kernel. The dispatch / stage /
resolver interface cost stays a sum (**M+N+K**), preventing combinatorial explosion.

**Why three axes — precision ⊥ backend are separable**:
If format (precision) and hardware (backend) were the same concept it would be two axes; if
separable, three. The code proves the separation — a single OpenCL backend holds 6
precision kernels (f32/f16/q4_0/q8_0/q6_k/mxfp4), and q4_0 exists on all three backends
(CPU/GPU/CUDA). The mapping is many-to-many, so precision is not a function of backend → an
independent axis.

**Composite operations = compositions of single-axis primitives**:
Several real operations touch two axes at once — weight swap (stage load + format precision
change), switch+migrate (hardware move), KIVI (stage modify + format precision). These
decompose into *compositions of single-axis primitives*. A new composite feature = a new
combination of existing primitives → 0 lines of new monolith. Orthogonality holds at the
primitive level, and a composite operation becomes a vector in the 3-space.

**(format × hardware) kernels are M×N — but isolated below the Backend interface**:
A q4-on-Adreno kernel and a q4-on-CPU kernel are physically different code, so the (format ×
hardware) kernel matrix is an irreducible M×N. But that matrix is confined to the dispatch
inside `backend.matmul(q4_tensor)` — callers, stages, and the resolver never know which
kernel fires. Confining the unavoidable M×N to a single leaf is the heart of the 3-axis
structure: the point where you get performance (specialized kernels) and extensibility
(additive interfaces) at once.

### stage — adjusting memory-resident data

**Stage**:
A cross-cutting action that intervenes at each phase while the decode loop runs to modify,
delete, or load *memory-resident data* (kvcache, weights). Most Stages adjust only the
presence/amount of data without changing its representation (format) or location (hardware)
— eviction, merge, weight swap. Execution ordering is the integrator's responsibility;
safety (crash-free) is the framework's.
_Avoid_: Handler, hook — those are just the current code's implementation names for the same
concept; the domain term is Stage. (Note: the code's `PipelineStage` *mechanism* name and
the *axis* name "stage" live at different levels — → Flagged ambiguities.)

**KVCacheStage** (the stage-axis trait):
The abstraction trait for the stage-axis behavior of deciding *how to adjust* the resident
tokens of the KV cache. It looks at the cache state + scores and produces a **`KVCachePlan`**
(tokens to `keep` + weighted `merge`); the buffer mutation is performed by the engine
executing that plan (plan-returning — the plugin decides, the engine executes). **The
sibling of `KVCacheFormat`** — `KVCacheFormat` = stored *representation* (precision/layout)
⊥ `KVCacheStage` = resident *token adjustment* (format ⊥ stage axes). Members: `Sliding` /
`H2O` / `Streaming` (layer-wide keep), `H2O+` (per-head keep), `D2O` (weighted merge). A new
stage = a separate technique crate + one line of registration (0 edits to engine core).

**KVCachePlan** (the output of a KVCacheStage):
The specification a stage produces of *what state this KV cache should become*. Composed of
`keep` (tokens to retain — layer-wide or per-head) + `merge` (weighted-merge instructions).
The engine executes it via `compact`. Named at KVCache scope because it covers keep+merge,
not just "eviction."

**EvictionPolicy** (legacy, being absorbed into KVCacheStage):
The old in-place eviction trait — `evict(&mut KVCache)` mutates the buffer *directly*.
Being evict-only, it cannot express merge (D2O) or per-head (H2O+). It is being generalized
into and replaced by the plan-returning `KVCacheStage` (they coexist during migration). Its
internal notion of "which tokens to drop" is absorbed into the `KVCacheStage::plan` decision.

### format — computation representation

**format**:
The *representation in which* KV or weight data is stored and computed. It is the data
coordinate of "in which format we compute." One concrete representation is one format member
— e.g. `f16`, `q4_0`, `KIVI` (Q2 packed + F32 residual). Adding a new format brings a new
byte layout + a dedicated kernel (**paired kernel**, per backend) — a heavy but isolated
extension.
_Note_: precision (q4/f16) and byte layout are not two separable concepts but one coordinate.
(The earlier model separated them, claiming "precision is a parameter inside Format"; that
was dropped — q4 and f16 already use different paired kernels, so precision is exactly what
splits the representation.)
_Avoid_: Layer (→ Flagged ambiguities). Precision (it is just another name for format, not a
separate axis).

**KVCacheFormat / WeightFormat**:
The base trait abstracting the various formats. To callers it exposes only geometry
(position/capacity), mutation (write/compact), and attention (KV) or dispatch (weight),
hiding internals like dtype/codebook/precision (format-agnostic). Stages manipulate data
through this trait.

**Paired kernel**:
A dedicated kernel that reads a format's byte layout to perform a computation. A different
format reads with a different kernel, and it must also differ per backend, so they form a
(format × hardware) M×N matrix — the irreducible leaf cost that comes with adding a new
format/backend, isolated below the Backend interface.

### hardware — compute location

**hardware (compute location)**:
The physical resource a computation runs on. It decomposes into two orthogonal
sub-dimensions — **compute** (the compute unit = backend: CPU NEON / Adreno OpenCL / Jetson
CUDA / NPU) and **data** (the memory = memory allocator). The two are orthogonal: on UMA
(ARM SoCs) several backends share one memory (only the compute unit changes), while on a
discrete GPU each backend has its own memory (VRAM↔RAM moves). That is why (backend, memory)
is not bound 1:1.

**Hardware** (the runtime resolver):
The runtime object that interprets hardware-axis coordinates. Internally it holds a backend
registry ⊥ a memory registry separately, and `resolve(target)` confines the UMA/discrete
branch of "to reach this backend, which memory?" to one place. It is a **read-only
resolver** — it does not hold the active backend mutably ("current device" is decode-loop
local state), and no computation observes "the current one" through this object (execution
gets the backend from storage via the tensor's tag).

**switch** (a hardware-axis move):
Switches the compute location GPU↔CPU (migrating KV to the new backend's memory if needed)
without changing the representation (format). It runs only at safe boundaries (e.g.
prefill→decode). It is the operation that moves data along the hardware axis. In the 3-axis
model, switch is a hardware-axis operation and is no longer a Stage.

**partition** (a format × hardware product):
Distributing one layer's forward across several backends concurrently. Each slice can have a
different (format, hardware) coordinate — e.g. the GPU slice is f16 and the NPU slice is q4
(HeteroLLM). So partition = the product of format (representation) × hardware (location), and
the target backends are obtained from `Hardware.resolve()`. A single "current location"
pointer cannot stamp two coordinates at once, which is exactly why the resolver is needed.

**Pressure / PressureSource** (graded system-pressure input):
An input signal that **fuses** several system pressures (memory, thermal, energy, …) **into a
single 0–100 scalar (`Pressure`)**, plus the pluggable source that supplies that value. It is
not an axis — a Stage reads it and decides a *graded response* (e.g. eviction strength).
**Intentional lossy unification**: consuming Stages do not distinguish the *origin* of the
pressure (memory/thermal/…) — origin loss is accepted for simplicity and manageability, so
the carrier is a single scalar and extending the signal *kinds* (anymap) is unnecessary (a
closed core vocabulary). **However, *mode* actions that cannot be reduced to a scalar**
(switch/suspend = on/off — there is no "switch by 73") flow not through this scalar but
through a separate *discrete command channel* (→ `EngineCommand`): Pressure carries only the
*graded input* and mode outputs are kept separate. `ManagerPressureSource` (the manager's
aggregated value) / `LocalPressureSource` (manager-less autonomous, fusing all sensors) /
third parties all hide behind the same `fn pressure(&self) -> Pressure`, and the consuming
Stage does not know which source it is. If an origin-discriminating policy is ever needed,
reconsider a source tag (a promotion trigger). The source is held at construction (the
swap point); the value is read-only per step via `StepInfo`.

### How a Stage holds its data — the 3 handle kinds

The static type of the reference by which a Stage holds the data (format) it will adjust. An
unordered menu of 3; there is no fourth.

**Base-trait-handle**:
An `Arc<dyn KVCacheFormat>`. A Stage holding this *must not* know which format it is. Example:
Sliding, which looks only at position.

**Concrete-handle**:
A handle to a specific format type, like `Arc<StandardFormat>`. Knowing that type *is normal*
(that is what holding a concrete means). Example: D2O, which must read the original K
directly.

**Capability-handle**:
An `Arc<dyn SomeCapability>`. It abstracts a *capability* that cuts across heterogeneous
formats. Create one only when there are two or more consumers (a single consumer is a
hypothetical seam, so you do not create it).

## Flagged ambiguities

- **Layer is for the transformer layer only**: "Layer" (`LlamaLayer`/`TransformerLayer`, the
  16 decoder blocks, `LayerSlot`·`layer_idx`, etc.) refers to *model structure* only. The
  *representation* of KV/weight data is never called "Layer" but **format**
  (`KVCacheFormat`/`WeightFormat`).
- **stage axis vs the `PipelineStage` mechanism vs the `KVCacheStage` trait**: "stage" (the
  axis) is the *conceptual dimension* of adjusting memory-resident data; the code's
  `PipelineStage`/`CachePressureHandler` are the decode-loop hook *mechanism*; and
  **`KVCacheStage` is the formal trait** (plan-returning) by which a plugin implements that
  stage-axis behavior. The three live at different levels. **The session
  `EvictionStage`/`SwapStage` (`session/traits.rs`) are deprecated** — a dormant abstraction
  that tried to make the WHEN-hook a plugin; WHEN (decode-loop integration) is owned by the
  engine and WHAT (the keep/merge decision) is split into `KVCacheStage`, so it was dropped.
- **format is the representation (precision+layout fused)**: the old "precision is a parameter
  inside Format / Standard and KIVI are separate byte layouts" is dropped. q4/f16/KIVI are
  each one coordinate of the format axis.
- **device is now an axis (the old "substrate" is dropped)**: the code confirms precision
  (format) and backend (hardware) are separable (a single backend holds 6 precisions, q4
  exists on 3 backends) → hardware is a formal axis. switch = a hardware-axis operation,
  partition = a format × hardware product. (The M×N kernel matrix is isolated below the
  Backend interface, so axis additivity is preserved.)
- **Eviction is a stage and KIVI is a format (+ a stage composition)**: eviction is
  *resident-data adjustment* that drops tokens, so it is a **stage**. KIVI quantization
  changes the byte representation (a *format*) and its conversion action *modifies resident
  data*, so it is a **composition** with a stage.

## Example dialogue

> **Dev**: I want to add KIVI — can I just slot it in as a stage like Sliding or H2O?
> **Expert**: Only partly. Sliding/H2O are **stage** actions of "which tokens to drop," so they
> don't touch the representation. KIVI *re-lays* KV as Q2 (modifying resident data = stage) and
> then runs attention at Q2 precision on top of it (= format) — a **composition**. So it brings
> a KVCacheFormat impl + a paired kernel (format).
> **Dev**: Then can I run Sliding on top of the KIVI format?
> **Expert**: Yes. That is exactly what it means for the three axes to be **orthogonal**. On top
> of KIVI (a format coordinate), Sliding (a stage action) just calls `compact`. Sliding doesn't
> know whether what's underneath is Q2 or F32 — because it holds a **base-trait-handle**.
> **Dev**: Which axis is the switch that moves something running on GPU over to CPU?
> **Expert**: The **hardware** axis. It moves the compute *location*, so it changes neither the
> representation (format) nor the behavior (stage) — KIVI is KIVI whether on GPU or CPU. The
> switch migrating KV to a new memory is also part of that hardware-axis move.
> **Dev**: In partition I want the GPU at f16 and the NPU at q4.
> **Expert**: That is a format × hardware product. Each slice gets a different coordinate, (f16,
> GPU) / (q4, NPU). A single "current device" pointer can't stamp two coordinates at once, so
> you get the per-slice backend from `Hardware.resolve()`.
> **Dev**: Is that "format" different from a transformer layer?
> **Expert**: Completely. A transformer **layer** is a model decoder block (16 of them), while
> **format** is *in what representation* those layers' KV/weight are computed.
