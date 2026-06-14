//! `RpcmemSecondaryStore` — LISWAP-6 DMA-BUF alias backing for secondary GGUF
//! weights when the primary backend is `qnn_oppkg`.
//!
//! ## 동기
//!
//! LISWAP-5 v1 fair comparison (postmortem §6, §7.6) 결과 swap copy 비용 (~480 ms
//! blocking) 이 단일-shot, per-tick=1, phase-aware 모든 모드에서 핵심 cost 임이
//! 확인되었다. `qnn_oppkg` backend의 KV cache가 이미 rpcmem heap + OpenCL
//! `CL_MEM_USE_HOST_PTR` alias로 zero-copy 가속을 검증했으므로 (`hybrid_memory.rs::android_alloc_kv`),
//! 본 모듈은 동일 패턴을 secondary weight 로 확장하여 H2D copy 자체를 제거한다.
//!
//! ## 설계 — Lazy per-layer alloc
//!
//! Model load 시점에 rpcmem alloc을 하지 않는다. 첫 swap target layer 가 접근될
//! 때 (`ensure_layer_loaded`) 그 layer 의 모든 tensor 를 단일 rpcmem region 으로
//! alloc + memcpy from mmap. 한 번 alloc 된 layer region 은 model lifetime 까지
//! 유지되어 두 번째 swap 부터는 cache hit 으로 alias 만 새로 만든다.
//!
//! 메모리 피크 분석:
//! - 전체 swap 진행 시: 25 layer × ~50 MB = ~1.25 GB peak (Galaxy S25 12 GB RAM 에서 OK)
//! - swap 일부만 진행: 해당 layer 만 alloc → 피크 감소
//! - swap 미발생: rpcmem 0 byte
//!
//! ## Lifetime / Drop
//!
//! - `RpcmemLayerRegion::Drop` 이 `rpcmem_free(host_ptr)` 호출 → 해당 region 의
//!   모든 외부 cl_mem alias 가 먼저 drop 된 이후에야 안전.
//! - alias buffer 는 `Arc<SecondaryMmap>` 보유 → secondary store 가 살아있는 동안
//!   region map 도 유지 → region drop 은 store drop 시점까지 지연.
//!
//! Spec: ENG-DAT-094 (SecondaryMmap), INV-143 (alias lifetime), ENG-QNN-204
//! (rpcmem heap reuse).

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::backend::Backend;
use crate::buffer::{Buffer, DType};
use crate::memory::rpcmem::allocator::RpcmemAllocator;
use crate::model_config::ModelConfig;
use crate::models::loader::gguf::GgufFile;
#[cfg(target_os = "android")]
use crate::models::loader::gguf::{ggml_type_to_dtype, parse_model_config, tensor_byte_size};
use crate::models::weights::SecondaryMmap;
use crate::models::weights::backing::WeightSectionView;
#[cfg(target_os = "android")]
use crate::models::weights::backing::{AufBacking, GgufBacking};
#[cfg(target_os = "android")]
use crate::models::weights::secondary_mmap::parse_block_tensor_name;
use crate::models::weights::secondary_mmap::{LayerTensorSlice, SecondaryTensorInfo};

/// Per-layer rpcmem region. Owns one `rpcmem_alloc` allocation that holds
/// every swap-relevant tensor for the layer, with `tensor_map` describing
/// each tensor's offset/length within the region.
///
/// Sprint 2a Phase 2: `Arc<RpcmemAllocator>` 보유로 lifetime 을 type system
/// 강제 (INV-RPCMEM-005). Drop 순서: region 먼저, allocator 나중.
pub struct RpcmemLayerRegion {
    /// Base host pointer returned by `RpcmemAllocator::alloc` (mmap'd DMA-BUF).
    host_ptr: *mut u8,
    /// Total bytes allocated. Read by Drop diagnostics + size sanity tests.
    /// Allowed to be `dead_code` on non-Android because the Drop arm is gated.
    #[allow(dead_code)]
    size: usize,
    /// Subname (e.g. "attn_q.weight") → (offset_in_region, byte_len, dtype).
    tensor_map: HashMap<String, (usize, usize, DType)>,
    /// rpcmem allocator — Arc 보유로 buffer lifetime ⊂ allocator lifetime
    /// 을 type system 으로 강제 (INV-RPCMEM-005). Drop 에서 `allocator.free`
    /// 호출. host 빌드에서도 Arc field 자체는 보유 가능 (Drop 분기에서
    /// no-op 처리).
    allocator: Arc<RpcmemAllocator>,
}

// SAFETY: rpcmem_alloc returns a host-accessible mapping. The underlying
// memory is single-owner (this struct) and protected by `Mutex` at the
// `RpcmemSecondaryStore` level for HashMap insertions. fn-pointer is `Copy`
// and stateless. Drop runs at most once.
unsafe impl Send for RpcmemLayerRegion {}
unsafe impl Sync for RpcmemLayerRegion {}

// Step 3-E (V-09): L2 `memory/rpcmem/opencl_alias.rs::RpcmemAliasBuffer`
// holds an `Arc<dyn RpcmemRegionGuard>` purely as a drop-ordering anchor so
// that `rpcmem_free` runs only after the cl_mem alias is dropped (INV-143).
// Marker trait — no methods are invoked through the trait object.
impl crate::memory::secondary::RpcmemRegionGuard for RpcmemLayerRegion {}

impl RpcmemLayerRegion {
    /// Resolve a tensor's host pointer + length within this region.
    ///
    /// Returns `(host_ptr_base, offset, len, dtype)`. The caller adds
    /// `offset` to `host_ptr_base` to obtain the tensor's start address.
    pub fn tensor_info(&self, subname: &str) -> Option<(*mut u8, usize, usize, DType)> {
        let (offset, len, dtype) = self.tensor_map.get(subname).copied()?;
        Some((self.host_ptr, offset, len, dtype))
    }
}

/// Result of `RpcmemSecondaryStore::host_ptr_for` — bundles the lifetime
/// guard (`region` Arc) with the alias coordinates so the caller can hand
/// both to `Backend::alloc_alias_weight_buffer` without hitting the
/// "very complex type" clippy lint.
pub struct HostPtrAlias {
    pub region: Arc<RpcmemLayerRegion>,
    pub host_ptr: *mut u8,
    pub offset: usize,
    pub len: usize,
    pub dtype: DType,
}

impl Drop for RpcmemLayerRegion {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            // SAFETY: host_ptr produced by self.allocator.alloc; Drop runs
            // once; alias lifetime invariant guarantees no live cl_mem refs.
            // host build 에서는 ExternalFns variant 가 dummy 일 수 있으나
            // 호스트 path 는 host_ptr 가 null 이라 분기 진입 불가.
            unsafe { self.allocator.free(self.host_ptr) };
        }
    }
}

/// Lazy per-layer rpcmem-backed secondary weight store.
///
/// Owns the secondary GGUF mmap (used as the memcpy source for first-touch)
/// plus a `Mutex<HashMap>` of allocated per-layer regions. The store itself
/// does not allocate any rpcmem at construction time.
pub struct RpcmemSecondaryStore {
    /// Secondary weight backing — `WeightSectionView` trait object so the
    /// store can host either a GGUF mmap (`GgufBacking`) or an AUF self-secondary
    /// (`AufBacking`, shared `Arc<AufView>` with primary). `info.offset`은
    /// `backing.weights_bytes()`의 슬라이스 base 기준이라 두 경우 모두 같은
    /// 인덱싱 식 `bytes[info.offset..]`으로 통일된다.
    backing: Arc<dyn WeightSectionView>,
    /// Source path for diagnostics.
    source_path: PathBuf,
    /// Pre-parsed per-layer tensor index (subname → SecondaryTensorInfo).
    /// Used by `layer_tensor()` callers and by `ensure_layer_loaded` to know
    /// which tensors to memcpy into the rpcmem region.
    layer_index: Vec<LayerTensorSlice>,
    /// Per-layer rpcmem region cache. `Mutex` synchronises concurrent
    /// `ensure_layer_loaded` calls; lookups are O(1) once allocated.
    layer_regions: Mutex<HashMap<usize, Arc<RpcmemLayerRegion>>>,
    /// LISWAP-6 Phase 1 — pre-built `cl_mem` aliases keyed by `(layer, subname)`.
    /// Populated eagerly inside `ensure_layer_loaded` when both `backend_weak`
    /// and `self_weak` are installed. Subsequent `cached_alias()` lookups
    /// return the cached `Arc<dyn Buffer>` without going through the
    /// ~4.5 ms `clCreateBuffer(USE_HOST_PTR)` path.
    ///
    /// Cycle break: `RpcmemAliasBuffer` holds `Weak<SecondaryMmap>` (not
    /// strong) so this strong cache does not pin the enclosing
    /// `Arc<SecondaryMmap>` against itself. The graph drops as one when the
    /// model releases its `Arc<SecondaryMmap>`.
    alias_cache: Mutex<HashMap<(usize, String), Arc<dyn Buffer>>>,
    /// Backend used to create alias `cl_mem` handles. Weak so the model can
    /// release the backend independently; if the upgrade fails at populate
    /// time we silently skip caching (the swap path then falls back to the
    /// direct `alloc_alias_weight_buffer` call, preserving correctness).
    backend_weak: Mutex<Option<Weak<dyn Backend>>>,
    /// Weak self-reference back to the enclosing `Arc<SecondaryMmap>`. Set
    /// once via `install_self_arc` immediately after wrapping the store.
    /// The cached `RpcmemAliasBuffer` instances upgrade this to install
    /// their `Weak<SecondaryMmap>` lifetime back-reference.
    self_weak: OnceLock<Weak<SecondaryMmap>>,
    /// Sprint 2a Phase 2 (ENG-RPCMEM-030/032): backend-agnostic rpcmem
    /// allocator. `RpcmemLayerRegion` clones from this Arc so allocator
    /// lifetime ⊃ region lifetime (INV-RPCMEM-005). Shared with the
    /// OpenCLBackend via `EXT_RPCMEM_ALLOCATOR` extension (INV-RPCMEM-002).
    /// Android-only field: used in `build_layer_region` under `#[cfg(target_os = "android")]`.
    #[allow(dead_code)]
    allocator: Arc<RpcmemAllocator>,
}

impl std::fmt::Debug for RpcmemSecondaryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpcmemSecondaryStore")
            .field("source_path", &self.source_path)
            .field("num_layers", &self.layer_index.len())
            .field(
                "regions_loaded",
                &self.layer_regions.lock().map(|m| m.len()).unwrap_or(0),
            )
            .field(
                "aliases_cached",
                &self.alias_cache.lock().map(|m| m.len()).unwrap_or(0),
            )
            .finish()
    }
}

impl RpcmemSecondaryStore {
    /// Open a GGUF secondary file and prepare the lazy store.
    ///
    /// Sprint 2a Phase 2 (ENG-RPCMEM-030): `allocator` is the shared
    /// `Arc<RpcmemAllocator>` obtained from
    /// `OpenCLBackend::get_extension(EXT_RPCMEM_ALLOCATOR)` (or, during Sprint
    /// 2a coexistence, from `QnnOppkgRuntime` wrapped via
    /// `RpcmemAllocator::from_external_fns`). On non-Android targets this
    /// constructor returns an error — callers should fall back to the regular
    /// `SecondaryMmap::Gguf` variant.
    pub fn from_gguf(
        path: &std::path::Path,
        primary_config: &ModelConfig,
        primary_gguf: &GgufFile,
        backend: Arc<dyn Backend>,
        allocator: Arc<RpcmemAllocator>,
    ) -> Result<Self> {
        #[cfg(not(target_os = "android"))]
        {
            let _ = (path, primary_config, primary_gguf, backend, allocator);
            Err(anyhow!(
                "RpcmemSecondaryStore is Android-only (rpcmem heap unavailable on host targets)"
            ))
        }

        #[cfg(target_os = "android")]
        {
            let gguf = GgufFile::open(path)
                .map_err(|e| anyhow!("rpcmem_secondary: GGUF open failed: {e}"))?;

            // Validate metadata vs primary (same checks as open_secondary_gguf).
            let secondary_config = parse_model_config(&gguf)
                .map_err(|e| anyhow!("rpcmem_secondary: config parse failed: {e}"))?;
            check_metadata_match(primary_config, &secondary_config)?;

            // Build per-layer slice index (mirror of open_secondary_gguf logic).
            let num_layers = primary_config.num_hidden_layers;
            let mut layer_index: Vec<LayerTensorSlice> =
                vec![LayerTensorSlice::default(); num_layers];

            let tensor_data_offset = gguf.tensor_data_offset();
            for info in &gguf.tensors {
                let Ok(dtype) = ggml_type_to_dtype(info.ggml_type) else {
                    continue;
                };
                let byte_size = tensor_byte_size(info);
                if byte_size == 0 {
                    continue;
                }
                let slice_info = SecondaryTensorInfo {
                    offset: tensor_data_offset + info.offset as usize,
                    len: byte_size,
                    dtype,
                    dims: info.dims.clone(),
                };
                if let Some((layer_idx, subname)) = parse_block_tensor_name(&info.name) {
                    if layer_idx >= num_layers {
                        continue;
                    }
                    layer_index[layer_idx]
                        .tensors
                        .insert(subname.to_string(), slice_info);
                }
            }

            // Validate decoder tensor presence + shapes against the primary.
            for info in &primary_gguf.tensors {
                let Some((layer_idx, subname)) = parse_block_tensor_name(&info.name) else {
                    continue;
                };
                if layer_idx >= num_layers {
                    continue;
                }
                let Some(secondary_info) = layer_index[layer_idx].tensors.get(subname) else {
                    return Err(anyhow!(
                        "rpcmem_secondary: missing tensor '{}' in secondary file",
                        info.name
                    ));
                };
                if secondary_info.dims != info.dims {
                    return Err(anyhow!(
                        "rpcmem_secondary: shape mismatch for '{}': primary={:?}, secondary={:?}",
                        info.name,
                        info.dims,
                        secondary_info.dims
                    ));
                }
            }

            let backing: Arc<dyn WeightSectionView> = Arc::new(GgufBacking {
                gguf: Arc::new(gguf),
                source_path: path.to_path_buf(),
            });

            Ok(Self {
                backing,
                source_path: path.to_path_buf(),
                layer_index,
                layer_regions: Mutex::new(HashMap::new()),
                alias_cache: Mutex::new(HashMap::new()),
                backend_weak: Mutex::new(Some(Arc::downgrade(&backend))),
                self_weak: OnceLock::new(),
                allocator,
            })
        }
    }

    /// W-AUF-2.3 — AUF self-secondary 진입.
    ///
    /// 이미 `from_auf_self_secondary` (loader)로 빌드된 `(layer_index, Arc<AufView>)`를
    /// rpcmem alias 백킹으로 재포장한다. mmap 1회 + RpcMem 경로 동시 활성을 보장한다.
    ///
    /// 호스트 빌드에서는 에러 반환 (Android-only) — 호출자는 `SecondaryMmap::Auf`
    /// 일반 variant로 fallback해야 한다.
    pub fn from_auf_self_secondary(
        view: Arc<crate::auf::AufView>,
        layer_index: Vec<LayerTensorSlice>,
        diag_source_path: PathBuf,
        backend: Arc<dyn Backend>,
        allocator: Arc<RpcmemAllocator>,
    ) -> Result<Self> {
        #[cfg(not(target_os = "android"))]
        {
            let _ = (view, layer_index, diag_source_path, backend, allocator);
            Err(anyhow!(
                "RpcmemSecondaryStore::from_auf_self_secondary is Android-only \
                 (rpcmem heap unavailable on host targets); caller must fall back \
                 to SecondaryMmap::Auf"
            ))
        }

        #[cfg(target_os = "android")]
        {
            let backing: Arc<dyn WeightSectionView> = Arc::new(AufBacking {
                view,
                source_path: diag_source_path.clone(),
            });
            Ok(Self {
                backing,
                source_path: diag_source_path,
                layer_index,
                layer_regions: Mutex::new(HashMap::new()),
                alias_cache: Mutex::new(HashMap::new()),
                backend_weak: Mutex::new(Some(Arc::downgrade(&backend))),
                self_weak: OnceLock::new(),
                allocator,
            })
        }
    }

    /// Install the enclosing `Arc<SecondaryMmap>` as a weak self-reference.
    /// Called by `open_secondary_with_backend` immediately after wrapping
    /// the store in `Arc::new(SecondaryMmap::Rpcmem(...))`. Idempotent —
    /// subsequent calls are silent no-ops (`OnceLock::set` returns Err).
    ///
    /// Without this, `ensure_layer_loaded` cannot populate the alias cache
    /// (the `RpcmemAliasBuffer` constructor needs an `Arc<SecondaryMmap>`
    /// to downgrade). The swap path then falls back to direct
    /// `alloc_alias_weight_buffer` calls, preserving correctness at the cost
    /// of the per-chunk overhead the cache was built to eliminate.
    pub fn install_self_arc(&self, secondary: &Arc<SecondaryMmap>) {
        let _ = self.self_weak.set(Arc::downgrade(secondary));
    }

    /// Look up a pre-built alias by `(layer, subname)`. Returns `None` if the
    /// layer was never prefaulted, the tensor isn't in this layer, or the
    /// store was constructed without a backend.
    pub fn cached_alias(&self, layer_idx: usize, subname: &str) -> Option<Arc<dyn Buffer>> {
        let cache = self.alias_cache.lock().ok()?;
        cache.get(&(layer_idx, subname.to_string())).cloned()
    }

    /// Diagnostic — current size of the alias cache (used by the eager
    /// prefault log line in `generate.rs`).
    pub fn alias_cache_len(&self) -> usize {
        self.alias_cache.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Eagerly populate `alias_cache` for `layer_idx` after the rpcmem
    /// region has been allocated. Silent no-op when prerequisites are
    /// missing (no backend / no self_weak / non-OpenCL backend) — the swap
    /// path falls back to direct allocation in that case.
    #[cfg_attr(not(feature = "opencl"), allow(unused_variables, unused_mut))]
    fn populate_alias_cache_for_layer(&self, layer_idx: usize, region: &Arc<RpcmemLayerRegion>) {
        // Both prerequisites must be installed; otherwise skip caching.
        let Some(secondary_arc) = self.self_weak.get().and_then(Weak::upgrade) else {
            return;
        };
        let Some(backend) = self
            .backend_weak
            .lock()
            .ok()
            .and_then(|g| g.as_ref().and_then(Weak::upgrade))
        else {
            return;
        };

        // Snapshot tensor entries (stable iteration; no lock during alloc).
        let entries: Vec<(String, *mut u8, usize, usize, DType)> = {
            let mut v: Vec<_> = region
                .tensor_map
                .iter()
                .map(|(name, &(offset, len, dtype))| {
                    (name.clone(), region.host_ptr, offset, len, dtype)
                })
                .collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };

        let mut new_entries: Vec<((usize, String), Arc<dyn Buffer>)> =
            Vec::with_capacity(entries.len());
        for (subname, host_ptr, offset, len, dtype) in entries {
            #[cfg(feature = "opencl")]
            {
                // SAFETY: `host_ptr` is `region.host_ptr` returned by rpcmem
                // alloc; `offset..offset+len` lies within the layer's region
                // (built by `tensor_map` at scan time). Lifetime is pinned by
                // `secondary_arc` (whole store) + `Arc::clone(region)` (this
                // layer's rpcmem allocation), both moved into the alias
                // buffer per `Backend::alloc_alias_weight_buffer` contract.
                let secondary_dyn: Arc<dyn crate::memory::host::mmap::MmapKeepAlive> =
                    secondary_arc.clone();
                let region_dyn: Arc<dyn crate::memory::secondary::RpcmemRegionGuard> =
                    region.clone();
                let alloc_res = unsafe {
                    backend.alloc_alias_weight_buffer(
                        host_ptr,
                        offset,
                        len,
                        dtype,
                        secondary_dyn,
                        region_dyn,
                    )
                };
                match alloc_res {
                    Ok(Some(buf)) => {
                        new_entries.push(((layer_idx, subname), buf));
                    }
                    Ok(None) => {
                        // Backend declined (CPU, non-OpenCL, etc.) — leave the
                        // (layer, subname) entry uncached; swap path falls back.
                    }
                    Err(e) => {
                        eprintln!(
                            "[liswap6] alias cache populate failed (layer={layer_idx}, \
                             subname='{subname}'): {e} — fallback path will allocate at swap time"
                        );
                    }
                }
            }
            #[cfg(not(feature = "opencl"))]
            {
                let _ = (host_ptr, offset, len, dtype, &subname, &backend);
            }
        }

        if new_entries.is_empty() {
            return;
        }
        if let Ok(mut cache) = self.alias_cache.lock() {
            for (key, buf) in new_entries {
                cache.entry(key).or_insert(buf);
            }
        }
    }

    /// Source path (diagnostic).
    pub fn source_path(&self) -> &std::path::Path {
        &self.source_path
    }

    /// Look up a tensor descriptor by `(layer_idx, subname)`. Mirrors
    /// `GgufSecondaryMmap::layer_tensor`.
    pub fn layer_tensor(&self, layer_idx: usize, subname: &str) -> Option<&SecondaryTensorInfo> {
        self.layer_index
            .get(layer_idx)
            .and_then(|slice| slice.tensors.get(subname))
    }

    /// Zero-copy byte slice from the underlying backing. Used by fallback paths
    /// (qk-unpermute, GGUF/AUF source-of-truth) when the alias path is not viable.
    pub fn tensor_bytes(&self, info: &SecondaryTensorInfo) -> &[u8] {
        let end = info.offset + info.len;
        &self.backing.weights_bytes()[info.offset..end]
    }

    /// Ensure the rpcmem region for `layer_idx` is allocated and populated.
    ///
    /// First call: allocates a single `rpcmem_alloc` of size = sum(layer
    /// tensors), then memcpys each tensor from the mmap into the region.
    /// Subsequent calls: O(1) HashMap hit.
    ///
    /// Returns an `Arc` so callers can share the region across multiple
    /// alias buffers (one per tensor) without lifetime gymnastics.
    pub fn ensure_layer_loaded(&self, layer_idx: usize) -> Result<Arc<RpcmemLayerRegion>> {
        // Fast path: cache hit (no rpcmem alloc).
        {
            let regions = self
                .layer_regions
                .lock()
                .map_err(|_| anyhow!("rpcmem_secondary: layer_regions mutex poisoned"))?;
            if let Some(region) = regions.get(&layer_idx) {
                return Ok(region.clone());
            }
        }

        // Slow path: build the region. Drop the lock during memcpy / alloc
        // so concurrent `ensure_layer_loaded` for *different* layers can
        // proceed. We re-acquire after the build to insert.
        let region = self.build_layer_region(layer_idx)?;

        let installed = {
            let mut regions = self
                .layer_regions
                .lock()
                .map_err(|_| anyhow!("rpcmem_secondary: layer_regions mutex poisoned"))?;
            // Race: another thread may have built the same layer concurrently.
            // Use entry() to keep the existing one and discard ours.
            regions.entry(layer_idx).or_insert(region).clone()
        };

        // LISWAP-6 Phase 1 — eagerly create cl_mem aliases for every tensor
        // in this layer so the swap path's `cached_alias` lookup hits without
        // paying ~4.5 ms per `clCreateBuffer(USE_HOST_PTR)`. Idempotent: a
        // second prefault for the same layer hits a region cache + finds the
        // alias entries already present (entry().or_insert keeps the first).
        self.populate_alias_cache_for_layer(layer_idx, &installed);

        Ok(installed)
    }

    /// Resolve a tensor's host_ptr + offset + len + dtype, allocating the
    /// region lazily if needed. Used by the alias buffer construction path.
    ///
    /// Returns `Ok(None)` when the tensor is missing from the secondary
    /// (caller should surface a proper error or fall back).
    pub fn host_ptr_for(&self, layer_idx: usize, subname: &str) -> Result<Option<HostPtrAlias>> {
        let region = self.ensure_layer_loaded(layer_idx)?;
        let Some((base, offset, len, dtype)) = region.tensor_info(subname) else {
            return Ok(None);
        };
        Ok(Some(HostPtrAlias {
            region,
            host_ptr: base,
            offset,
            len,
            dtype,
        }))
    }

    /// Force-allocate and populate regions for the given target layers.
    /// Used by `prefault_layers` to amortize first-touch cost.
    pub fn prefault_layers(&self, target_layers: &[usize]) {
        for &layer_idx in target_layers {
            let _ = self.ensure_layer_loaded(layer_idx);
        }
    }

    // ── internal ─────────────────────────────────────────────────────────────

    #[cfg(target_os = "android")]
    fn build_layer_region(&self, layer_idx: usize) -> Result<Arc<RpcmemLayerRegion>> {
        let layer_slice = self.layer_index.get(layer_idx).ok_or_else(|| {
            anyhow!(
                "rpcmem_secondary: layer_idx {layer_idx} out of range (max {})",
                self.layer_index.len()
            )
        })?;

        // Stable iteration order so the offset map is deterministic across
        // builds (HashMap iteration order is not stable but we sort here for
        // diagnostic clarity; functional correctness does not require it).
        let mut entries: Vec<(&String, &SecondaryTensorInfo)> =
            layer_slice.tensors.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        // Compute total size + per-tensor offset map.
        let mut total: usize = 0;
        let mut tensor_map: HashMap<String, (usize, usize, DType)> =
            HashMap::with_capacity(entries.len());
        for (subname, info) in &entries {
            // Align each tensor to 64-byte boundary so OpenCL `CL_MEM_USE_HOST_PTR`
            // alias respects the host alignment requirement (typically 1-byte for
            // unified memory but 64 is the safe upper bound on Adreno).
            let aligned_offset = total.next_multiple_of(64);
            tensor_map.insert((*subname).clone(), (aligned_offset, info.len, info.dtype));
            total = aligned_offset + info.len;
        }

        if total == 0 {
            return Err(anyhow!(
                "rpcmem_secondary: layer {layer_idx} has zero total tensor bytes"
            ));
        }

        // 1. rpcmem_alloc — single backing region for the whole layer.
        // SAFETY: self.allocator is shared (INV-RPCMEM-002) and outlives every
        // region (INV-RPCMEM-005). Failure → caller `ensure_layer_loaded`
        // surfaces it; secondary loader treats it as
        // `SecondaryUnavailable` and falls back to GGUF mmap (INV-RPCMEM-003).
        let (host_ptr, _fd) = unsafe { self.allocator.alloc(total)? };

        // 2. memcpy each tensor from the backing weights section into the region.
        // Best-effort: if any copy fails (would only happen on a corrupt range),
        // free the region and bubble up.
        let mmap_bytes = self.backing.weights_bytes();
        for (subname, info) in &entries {
            let Some((dst_offset, len, _dt)) = tensor_map.get(*subname).copied() else {
                continue; // unreachable — we just inserted it
            };
            let src_start = info.offset;
            let src_end = info.offset + info.len;
            if src_end > mmap_bytes.len() || len != info.len {
                // Free + abort. SAFETY: host_ptr was just returned by
                // self.allocator.alloc; no aliases have been built yet.
                unsafe {
                    self.allocator.free(host_ptr);
                }
                return Err(anyhow!(
                    "rpcmem_secondary: tensor '{}' offset/len exceeds mmap (layer {layer_idx})",
                    subname
                ));
            }
            // SAFETY: `host_ptr` is `total` bytes; `dst_offset + len <= total`
            // by construction; `mmap_bytes[src_start..src_end]` is a valid
            // read-only mmap slice; src/dst do not overlap (different regions).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mmap_bytes[src_start..src_end].as_ptr(),
                    host_ptr.add(dst_offset),
                    len,
                );
            }
        }

        Ok(Arc::new(RpcmemLayerRegion {
            host_ptr,
            size: total,
            tensor_map,
            allocator: Arc::clone(&self.allocator),
        }))
    }

    #[cfg(not(target_os = "android"))]
    fn build_layer_region(&self, _layer_idx: usize) -> Result<Arc<RpcmemLayerRegion>> {
        Err(anyhow!(
            "rpcmem_secondary: build_layer_region called on non-Android target"
        ))
    }
}

/// Subset of `check_metadata` from `secondary_mmap.rs` reused locally.
/// We don't import the private `check_metadata` to avoid cross-module visibility
/// changes; instead we mirror the few invariants strictly required for swap.
///
/// Only invoked from the Android-gated `from_gguf` path; allowed to appear
/// dead on host targets where the constructor returns early.
#[allow(dead_code)]
fn check_metadata_match(primary: &ModelConfig, secondary: &ModelConfig) -> Result<()> {
    macro_rules! check {
        ($field:ident) => {
            if primary.$field != secondary.$field {
                return Err(anyhow!(
                    "rpcmem_secondary: metadata mismatch on '{}': primary={:?}, secondary={:?}",
                    stringify!($field),
                    primary.$field,
                    secondary.$field
                ));
            }
        };
    }
    check!(num_hidden_layers);
    check!(num_attention_heads);
    check!(num_key_value_heads);
    check!(hidden_size);
    check!(intermediate_size);
    check!(vocab_size);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // `RpcmemLayerRegion::size` is currently only consulted via Debug /
    // diagnostics. This compile-only probe keeps dead_code lint quiet across
    // feature combinations and documents the field's intended public surface.
    #[test]
    fn region_size_field_used() {
        let _check_field = |r: &RpcmemLayerRegion| -> usize { r.size };
    }
}
