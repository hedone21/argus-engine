//! SafeTensors-based tensor source for model loading.
//!
//! `SafetensorsSource` owns the mmap'd file data and provides tensor access
//! through the `TensorSource` trait. Supports single-file and multi-shard
//! safetensors layouts.

use anyhow::{Result, anyhow};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::backend::Backend;
// LAYER-EXEMPT: backend_concrete_downcast — §13.8-L
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::model_config::ModelConfig;
use crate::models::mappers::{WeightMapper, create_mapper_with_prefix};
use crate::shape::Shape;
use crate::tensor::Tensor;

use super::TensorId;
use super::convert;

/// Safetensors-backed tensor source.
///
/// Owns the mmap'd file data and parsed `SafeTensors` handles.
/// The `SafeTensors<'static>` lifetime is safe because the `Arc<Mmap>` in
/// `shard_mmaps` outlives the `SafeTensors` references (both are owned by
/// this struct and dropped together).
pub struct SafetensorsSource {
    config: ModelConfig,
    weight_dtype: DType,
    shard_mmaps: Vec<Arc<memmap2::Mmap>>,
    /// SAFETY: these SafeTensors reference data in `shard_mmaps`. The Arcs
    /// keep the mappings alive for the lifetime of this struct. We transmute
    /// the lifetime to 'static to store them, which is safe as long as we
    /// never hand out references that outlive `self`.
    shard_tensors: Vec<SafeTensors<'static>>,
    weight_map: HashMap<String, usize>,
    mapper: Box<dyn WeightMapper>,
    cpu_memory: Galloc,
    cpu_backend: Arc<CpuBackend>,
}

impl SafetensorsSource {
    /// Open a safetensors model directory.
    ///
    /// Supports both single-file (`model.safetensors`) and multi-shard layouts
    /// (`model.safetensors.index.json`).
    pub fn open(model_path: &str, weight_dtype: DType) -> Result<Self> {
        let path = Path::new(model_path);
        let config = ModelConfig::from_json(path)?;
        let mapper = create_mapper_with_prefix(config.arch, &config.weight_prefix);

        let single_path = path.join("model.safetensors");
        let index_path = path.join("model.safetensors.index.json");

        let (shard_files, weight_map): (Vec<String>, HashMap<String, usize>) =
            if single_path.exists() {
                (vec!["model.safetensors".to_string()], HashMap::new())
            } else if index_path.exists() {
                Self::parse_shard_index(&index_path)?
            } else {
                return Err(anyhow!(
                    "No model.safetensors or model.safetensors.index.json found in {}",
                    model_path
                ));
            };

        if shard_files.len() > 1 {
            eprintln!("Loading {} safetensors shards...", shard_files.len());
        }

        let shard_mmaps: Vec<Arc<memmap2::Mmap>> = shard_files
            .iter()
            .map(|f| {
                let file = File::open(path.join(f))?;
                let mmap = unsafe { MmapOptions::new().map(&file)? };
                #[cfg(target_os = "linux")]
                {
                    let ptr = mmap.as_ptr() as *mut libc::c_void;
                    let len = mmap.len();
                    unsafe {
                        libc::madvise(ptr, len, libc::MADV_HUGEPAGE);
                        libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
                    }
                }
                Ok(Arc::new(mmap))
            })
            .collect::<Result<_>>()?;

        let shard_tensors: Vec<SafeTensors<'static>> = shard_mmaps
            .iter()
            .map(|m| {
                let st = SafeTensors::deserialize(m.as_ref()).map_err(|e| anyhow!("{}", e))?;
                // SAFETY: The Arc<Mmap> in shard_mmaps keeps the backing memory alive
                // for the lifetime of this struct. We extend the borrow lifetime to
                // 'static because both shard_mmaps and shard_tensors are fields of
                // the same struct and are dropped together.
                Ok(unsafe { std::mem::transmute::<SafeTensors<'_>, SafeTensors<'static>>(st) })
            })
            .collect::<Result<_>>()?;

        Ok(Self {
            config,
            weight_dtype,
            shard_mmaps,
            shard_tensors,
            weight_map,
            mapper,
            cpu_memory: Galloc::new(),
            cpu_backend: Arc::new(CpuBackend::new()),
        })
    }

    /// Resolve a `TensorId` to the safetensors weight name string.
    pub fn resolve_name(&self, id: &TensorId) -> String {
        match id {
            TensorId::Embed => self.mapper.embed_name(),
            TensorId::FinalNorm => self.mapper.norm_name(),
            TensorId::LmHead => self.mapper.lm_head_name(),
            TensorId::LayerWeight { layer, kind } => {
                use super::LayerWeightKind;
                let names = self.mapper.weight_names(*layer);
                match kind {
                    LayerWeightKind::Wq => names.wq,
                    LayerWeightKind::Wk => names.wk,
                    LayerWeightKind::Wv => names.wv,
                    LayerWeightKind::Wo => names.wo,
                    LayerWeightKind::WGate => names.w_gate,
                    LayerWeightKind::WUp => names.w_up,
                    LayerWeightKind::WDown => names.w_down,
                    LayerWeightKind::AttentionNorm => names.attention_norm,
                    LayerWeightKind::FfnNorm => names.ffn_norm,
                    LayerWeightKind::PreFfnNorm => names.pre_ffn_norm.unwrap_or_default(),
                    LayerWeightKind::PostFfnNorm => names.post_ffn_norm.unwrap_or_default(),
                    LayerWeightKind::QNorm => names.q_norm.unwrap_or_default(),
                    LayerWeightKind::KNorm => names.k_norm.unwrap_or_default(),
                }
            }
            TensorId::LayerBias { layer, kind } => {
                use super::LayerBiasKind;
                let bias_names = self.mapper.bias_names(*layer);
                match (bias_names, kind) {
                    (Some(bn), LayerBiasKind::Bq) => bn.bq,
                    (Some(bn), LayerBiasKind::Bk) => bn.bk,
                    (Some(bn), LayerBiasKind::Bv) => bn.bv,
                    (None, _) => String::new(),
                }
            }
        }
    }

    /// Build a descriptive error for a missing tensor, including a sample of
    /// names that *are* present in the shard to aid debugging.
    fn missing_tensor_err(
        &self,
        name: &str,
        shard_idx: usize,
        inner: impl std::fmt::Display,
    ) -> anyhow::Error {
        let sample: Vec<&str> = self.shard_tensors[shard_idx]
            .names()
            .iter()
            .take(5)
            .copied()
            .collect();
        anyhow!(
            "safetensors missing tensor '{name}' in shard {shard_idx} \
             (config weight_prefix={:?}); first few names in shard: {sample:?}; inner: {inner}",
            self.config.weight_prefix
        )
    }

    /// Load a raw tensor by safetensors name into a GPU-backed tensor.
    ///
    /// `is_weight`: true uses `weight_dtype`, false uses F32.
    /// `finalize`: when the main backend is GPU, copies from CPU to GPU.
    fn load_raw(&self, name: &str, is_weight: bool, backend: &Arc<dyn Backend>) -> Result<Tensor> {
        let is_cpu = backend.name().contains("CPU");

        let finalize_tensor =
            |shape: Shape, buffer: Arc<dyn crate::buffer::Buffer>| -> Result<Tensor> {
                if is_cpu {
                    Ok(Tensor::new(shape, buffer, backend.clone()))
                } else {
                    let cpu_tensor =
                        Tensor::new(shape, buffer, self.cpu_backend.clone() as Arc<dyn Backend>);
                    // Weight tensors (`is_weight == true`) may take the
                    // backend's device-only weight path when enabled
                    // (e.g. `--cuda-weights-device`). Norms/biases are
                    // loaded via `load_raw(..., false, ...)` and stay on
                    // the zero-copy activation path.
                    if is_weight {
                        backend.copy_weight_from(&cpu_tensor)
                    } else {
                        backend.copy_from(&cpu_tensor)
                    }
                }
            };

        let shard_idx = self.weight_map.get(name).copied().unwrap_or(0);
        let tensor_view = match self.shard_tensors[shard_idx].tensor(name) {
            Ok(v) => v,
            Err(e) => return Err(self.missing_tensor_err(name, shard_idx, e)),
        };
        let shape = Shape::new(tensor_view.shape().to_vec());
        let num_elements: usize = tensor_view.shape().iter().product();
        let target_dtype = if is_weight {
            self.weight_dtype
        } else {
            DType::F32
        };

        // F16 weight path
        if target_dtype == DType::F16 {
            // Fast path: safetensors stores F16 -> reference mmap directly (0 copies)
            if tensor_view.dtype() == safetensors::Dtype::F16 && is_cpu {
                use crate::memory::host::mmap::MmapBuffer;
                let data = tensor_view.data();
                let data_offset =
                    data.as_ptr() as usize - self.shard_mmaps[shard_idx].as_ptr() as usize;
                let mmap_arc = self.shard_mmaps[shard_idx].clone();
                let buffer: Arc<dyn crate::buffer::Buffer> = Arc::new(unsafe {
                    MmapBuffer::new(mmap_arc, data_offset, data.len(), DType::F16)
                });
                return Ok(Tensor::new(shape, buffer, backend.clone()));
            }

            // Conversion path
            use half::f16;
            let size_bytes = num_elements * 2;
            let buffer = self.cpu_memory.alloc(size_bytes, DType::F16)?;
            let dst = unsafe {
                std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut f16, num_elements)
            };

            match tensor_view.dtype() {
                safetensors::Dtype::F16 => {
                    let data = tensor_view.data();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            buffer.as_mut_ptr(),
                            data.len(),
                        );
                    }
                }
                safetensors::Dtype::BF16 => {
                    convert::bf16_to_f16_buf(tensor_view.data(), dst, num_elements);
                }
                safetensors::Dtype::F32 => {
                    convert::f32_to_f16_buf(tensor_view.data(), dst, num_elements);
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported dtype in safetensors: {:?}",
                        tensor_view.dtype()
                    ));
                }
            }

            #[cfg(unix)]
            convert::release_source_pages(tensor_view.data());

            return finalize_tensor(shape, buffer);
        }

        // Q4_0 path
        if target_dtype == DType::Q4_0 {
            let mut f32_data = vec![0.0f32; num_elements];
            match tensor_view.dtype() {
                safetensors::Dtype::F32 => unsafe {
                    std::ptr::copy_nonoverlapping(
                        tensor_view.data().as_ptr(),
                        f32_data.as_mut_ptr() as *mut u8,
                        tensor_view.data().len(),
                    );
                },
                safetensors::Dtype::BF16 => {
                    convert::bf16_to_f32(tensor_view.data(), &mut f32_data, num_elements);
                }
                safetensors::Dtype::F16 => {
                    convert::f16_to_f32(tensor_view.data(), &mut f32_data, num_elements);
                }
                _ => {
                    return Err(anyhow!("Unsupported dtype: {:?}", tensor_view.dtype()));
                }
            }

            #[cfg(unix)]
            convert::release_source_pages(tensor_view.data());

            let rows = shape.dims()[0];
            let cols = shape.dims()[1];
            if !cols.is_multiple_of(crate::quant::QK4_0) {
                let buffer = self.cpu_memory.alloc(num_elements * 4, DType::F32)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        f32_data.as_ptr(),
                        buffer.as_mut_ptr() as *mut f32,
                        num_elements,
                    );
                }
                return finalize_tensor(shape, buffer);
            }

            let blocks = convert::quantize_q4_0(&f32_data, rows, cols);

            let size_bytes = blocks.len() * std::mem::size_of::<crate::quant::BlockQ4_0>();
            let buffer = self.cpu_memory.alloc(size_bytes, DType::Q4_0)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    blocks.as_ptr(),
                    buffer.as_mut_ptr() as *mut crate::quant::BlockQ4_0,
                    blocks.len(),
                );
            }
            return finalize_tensor(shape, buffer);
        }

        // F32 path (norms, embeddings)
        let buffer = self.cpu_memory.alloc(num_elements * 4, DType::F32)?;
        let dst = unsafe {
            std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut f32, num_elements)
        };

        match tensor_view.dtype() {
            safetensors::Dtype::F32 => unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor_view.data().as_ptr(),
                    buffer.as_mut_ptr(),
                    tensor_view.data().len(),
                );
            },
            safetensors::Dtype::BF16 => {
                convert::bf16_to_f32(tensor_view.data(), dst, num_elements);
            }
            safetensors::Dtype::F16 => {
                convert::f16_to_f32(tensor_view.data(), dst, num_elements);
            }
            _ => {
                return Err(anyhow!("Unsupported dtype: {:?}", tensor_view.dtype()));
            }
        }

        #[cfg(unix)]
        convert::release_source_pages(tensor_view.data());

        finalize_tensor(shape, buffer)
    }

    /// Load a raw tensor by safetensors name into a CPU-only tensor.
    fn load_raw_cpu(&self, name: &str, is_weight: bool) -> Result<Tensor> {
        let shard_idx = self.weight_map.get(name).copied().unwrap_or(0);
        let tensor_view = match self.shard_tensors[shard_idx].tensor(name) {
            Ok(v) => v,
            Err(e) => return Err(self.missing_tensor_err(name, shard_idx, e)),
        };
        let shape = Shape::new(tensor_view.shape().to_vec());
        let num_elements: usize = tensor_view.shape().iter().product();
        let target_dtype = if is_weight {
            self.weight_dtype
        } else {
            DType::F32
        };

        // F16 weight path: prefer MmapBuffer for zero-copy on CPU
        if target_dtype == DType::F16 {
            if tensor_view.dtype() == safetensors::Dtype::F16 {
                use crate::memory::host::mmap::MmapBuffer;
                let data = tensor_view.data();
                let data_offset =
                    data.as_ptr() as usize - self.shard_mmaps[shard_idx].as_ptr() as usize;
                let mmap_arc = self.shard_mmaps[shard_idx].clone();
                let buffer: Arc<dyn crate::buffer::Buffer> = Arc::new(unsafe {
                    MmapBuffer::new(mmap_arc, data_offset, data.len(), DType::F16)
                });
                return Ok(Tensor::new(
                    shape,
                    buffer,
                    self.cpu_backend.clone() as Arc<dyn Backend>,
                ));
            }

            // Conversion into Galloc buffer
            use half::f16;
            let size_bytes = num_elements * 2;
            let buffer = self.cpu_memory.alloc(size_bytes, DType::F16)?;
            let dst = unsafe {
                std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut f16, num_elements)
            };
            match tensor_view.dtype() {
                safetensors::Dtype::F16 => {
                    let data = tensor_view.data();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            buffer.as_mut_ptr(),
                            data.len(),
                        );
                    }
                }
                safetensors::Dtype::BF16 => {
                    convert::bf16_to_f16_buf(tensor_view.data(), dst, num_elements);
                }
                safetensors::Dtype::F32 => {
                    convert::f32_to_f16_buf(tensor_view.data(), dst, num_elements);
                }
                _ => {
                    return Err(anyhow!(
                        "Unsupported dtype in safetensors: {:?}",
                        tensor_view.dtype()
                    ));
                }
            }
            #[cfg(unix)]
            convert::release_source_pages(tensor_view.data());
            return Ok(Tensor::new(
                shape,
                buffer,
                self.cpu_backend.clone() as Arc<dyn Backend>,
            ));
        }

        // F32 path
        let buffer = self.cpu_memory.alloc(num_elements * 4, DType::F32)?;
        let dst = unsafe {
            std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut f32, num_elements)
        };
        match tensor_view.dtype() {
            safetensors::Dtype::F32 => unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor_view.data().as_ptr(),
                    buffer.as_mut_ptr(),
                    tensor_view.data().len(),
                );
            },
            safetensors::Dtype::BF16 => {
                convert::bf16_to_f32(tensor_view.data(), dst, num_elements);
            }
            safetensors::Dtype::F16 => {
                convert::f16_to_f32(tensor_view.data(), dst, num_elements);
            }
            _ => {
                return Err(anyhow!("Unsupported dtype: {:?}", tensor_view.dtype()));
            }
        }
        #[cfg(unix)]
        convert::release_source_pages(tensor_view.data());
        Ok(Tensor::new(
            shape,
            buffer,
            self.cpu_backend.clone() as Arc<dyn Backend>,
        ))
    }

    /// Check if a safetensors name exists in the loaded shards.
    fn has_raw(&self, name: &str) -> bool {
        if self.weight_map.is_empty() {
            self.shard_tensors[0].names().contains(&name)
        } else {
            self.weight_map.contains_key(name)
        }
    }

    /// Parse `model.safetensors.index.json`.
    fn parse_shard_index(index_path: &Path) -> Result<(Vec<String>, HashMap<String, usize>)> {
        #[derive(Deserialize)]
        struct ShardIndex {
            weight_map: HashMap<String, String>,
        }
        let index: ShardIndex = serde_json::from_reader(File::open(index_path)?)?;

        let mut shard_files: Vec<String> = Vec::new();
        let mut file_to_idx: HashMap<String, usize> = HashMap::new();
        let mut weight_map: HashMap<String, usize> = HashMap::new();

        for (tensor_name, shard_file) in &index.weight_map {
            let idx = if let Some(&i) = file_to_idx.get(shard_file) {
                i
            } else {
                let i = shard_files.len();
                shard_files.push(shard_file.clone());
                file_to_idx.insert(shard_file.clone(), i);
                i
            };
            weight_map.insert(tensor_name.clone(), idx);
        }

        eprintln!(
            "Shard index: {} tensors across {} shards",
            weight_map.len(),
            shard_files.len()
        );

        Ok((shard_files, weight_map))
    }
}

// -- TensorSource implementation --

impl super::TensorSource for SafetensorsSource {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weight_dtype(&self) -> DType {
        self.weight_dtype
    }

    fn load_tensor(
        &self,
        id: &TensorId,
        is_weight: bool,
        backend: &Arc<dyn Backend>,
        _memory: &dyn Memory,
    ) -> Result<Tensor> {
        let name = self.resolve_name(id);
        if name.is_empty() {
            return Err(anyhow!(
                "Tensor {:?} not available for this architecture",
                id
            ));
        }
        self.load_raw(&name, is_weight, backend)
    }

    fn load_tensor_cpu(
        &self,
        id: &TensorId,
        is_weight: bool,
        _memory: &dyn Memory,
    ) -> Result<Tensor> {
        let name = self.resolve_name(id);
        if name.is_empty() {
            return Err(anyhow!(
                "Tensor {:?} not available for this architecture",
                id
            ));
        }
        self.load_raw_cpu(&name, is_weight)
    }

    fn has_tensor(&self, id: &TensorId) -> bool {
        let name = self.resolve_name(id);
        if name.is_empty() {
            return false;
        }
        self.has_raw(&name)
    }

    fn cpu_backend(&self) -> Arc<dyn Backend> {
        self.cpu_backend.clone() as Arc<dyn Backend>
    }
}
