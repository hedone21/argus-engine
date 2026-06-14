//! GGUF v3 format parser and `GgufSource` (TensorSource impl).
//!
//! Self-contained parser for the GGUF binary format. No external GGUF crates.
//! Supports ggml_type: F32 (0), F16 (1), Q4_0 (2), Q4_1 (3), Q8_0 (8).
//! K-quants and IQ types are rejected with a clear error message.

use anyhow::{Result, anyhow, ensure};
use memmap2::MmapOptions;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::backend::Backend;
// LAYER-EXEMPT: backend_concrete_downcast — §13.8-L
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::memory::Memory;
use crate::memory::host::mmap::MmapBuffer;
use crate::model_config::ModelConfig;
use crate::shape::Shape;
use crate::tensor::Tensor;

use crate::memory::host::shared::SharedBuffer;

use super::convert::{dequant_q4_1, dequant_q4_k};
use super::{LayerBiasKind, LayerWeightKind, TensorId, TensorSource};

// ---------------------------------------------------------------------------
// GGUF constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x4655_4747; // 'GGUF' in LE: bytes [0x47, 0x47, 0x55, 0x46]
const GGUF_VERSION_3: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// ggml_type values
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q4_1: u32 = 3;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q4_K: u32 = 12;

// GGUF value types
const GGUF_TYPE_U8: u32 = 0;
const GGUF_TYPE_I8: u32 = 1;
const GGUF_TYPE_U16: u32 = 2;
const GGUF_TYPE_I16: u32 = 3;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_U64: u32 = 10;
const GGUF_TYPE_I64: u32 = 11;
const GGUF_TYPE_F64: u32 = 12;

// ---------------------------------------------------------------------------
// GGUF value enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

// ---------------------------------------------------------------------------
// GGUF tensor info
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Dimensions in GGUF order (innermost first).
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Byte offset relative to tensor data section start.
    pub offset: u64,
}

// ---------------------------------------------------------------------------
// Binary cursor (little-endian reads over a byte slice)
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(
            self.pos + n <= self.data.len(),
            "GGUF: unexpected EOF at offset {} (need {} bytes, {} remaining)",
            self.pos,
            n,
            self.remaining()
        );
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_bytes(1)?[0] as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    /// Read a GGUF string: u64 length + raw bytes (no null terminator).
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Read a GGUF value of the given type tag.
    fn read_value(&mut self, type_tag: u32) -> Result<GgufValue> {
        match type_tag {
            GGUF_TYPE_U8 => Ok(GgufValue::U8(self.read_u8()?)),
            GGUF_TYPE_I8 => Ok(GgufValue::I8(self.read_i8()?)),
            GGUF_TYPE_U16 => Ok(GgufValue::U16(self.read_u16()?)),
            GGUF_TYPE_I16 => Ok(GgufValue::I16(self.read_i16()?)),
            GGUF_TYPE_U32 => Ok(GgufValue::U32(self.read_u32()?)),
            GGUF_TYPE_I32 => Ok(GgufValue::I32(self.read_i32()?)),
            GGUF_TYPE_F32 => Ok(GgufValue::F32(self.read_f32()?)),
            GGUF_TYPE_BOOL => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            GGUF_TYPE_STRING => Ok(GgufValue::String(self.read_string()?)),
            GGUF_TYPE_ARRAY => {
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
            GGUF_TYPE_U64 => Ok(GgufValue::U64(self.read_u64()?)),
            GGUF_TYPE_I64 => Ok(GgufValue::I64(self.read_i64()?)),
            GGUF_TYPE_F64 => Ok(GgufValue::F64(self.read_f64()?)),
            _ => Err(anyhow!("GGUF: unknown value type tag {}", type_tag)),
        }
    }

    /// Align position to the given alignment boundary.
    fn align_to(&mut self, alignment: usize) {
        let rem = self.pos % alignment;
        if rem != 0 {
            self.pos += alignment - rem;
        }
    }
}

// ---------------------------------------------------------------------------
// GgufFile
// ---------------------------------------------------------------------------

pub struct GgufFile {
    mmap: Arc<memmap2::Mmap>,
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Name -> index into `tensors`.
    tensor_index: HashMap<String, usize>,
    /// Absolute byte offset of the tensor data section within the file.
    tensor_data_offset: usize,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("version", &self.version)
            .field("tensor_count", &self.tensors.len())
            .field("metadata_count", &self.metadata.len())
            .field("tensor_data_offset", &self.tensor_data_offset)
            .finish()
    }
}

impl GgufFile {
    /// Open and parse a GGUF file.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| anyhow!("Cannot open GGUF file {}: {}", path.display(), e))?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        #[cfg(target_os = "linux")]
        {
            let ptr = mmap.as_ptr() as *mut libc::c_void;
            let len = mmap.len();
            unsafe {
                libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
            }
        }

        let mmap = Arc::new(mmap);
        Self::parse(mmap)
    }

    /// Parse from already-mmap'd data (useful for testing).
    fn parse(mmap: Arc<memmap2::Mmap>) -> Result<Self> {
        // Clone the Arc before borrowing; the clone is cheap (ref count bump).
        let mmap_arc = mmap.clone();
        Self::parse_bytes(mmap.as_ref(), mmap_arc)
    }

    /// Internal: parse GGUF from a byte slice, retaining `mmap_arc` for lifetime.
    fn parse_bytes(data: &[u8], mmap_arc: Arc<memmap2::Mmap>) -> Result<Self> {
        let mut cur = Cursor::new(data);

        // Header
        let magic = cur.read_u32()?;
        ensure!(
            magic == GGUF_MAGIC,
            "GGUF: invalid magic number 0x{:08X} (expected 0x{:08X})",
            magic,
            GGUF_MAGIC
        );

        let version = cur.read_u32()?;
        ensure!(
            version == GGUF_VERSION_3,
            "GGUF: unsupported version {} (only v3 supported)",
            version
        );

        let tensor_count = cur.read_u64()? as usize;
        let kv_count = cur.read_u64()? as usize;

        // KV metadata
        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = cur.read_string()?;
            let type_tag = cur.read_u32()?;
            let value = cur.read_value(type_tag)?;
            metadata.insert(key, value);
        }

        // Read alignment from metadata (default 32)
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(a)) => *a as usize,
            Some(GgufValue::U64(a)) => *a as usize,
            _ => GGUF_DEFAULT_ALIGNMENT,
        };

        // Tensor infos
        let mut tensors = Vec::with_capacity(tensor_count);
        let mut tensor_index = HashMap::with_capacity(tensor_count);
        for i in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cur.read_u64()?);
            }
            let ggml_type = cur.read_u32()?;
            let offset = cur.read_u64()?;

            tensor_index.insert(name.clone(), i);
            tensors.push(GgufTensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        // Align to tensor data section
        cur.align_to(alignment);
        let tensor_data_offset = cur.pos;

        Ok(Self {
            mmap: mmap_arc,
            version,
            metadata,
            tensors,
            tensor_index,
            tensor_data_offset,
        })
    }

    /// Find a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Hint the OS that the mmap pages are no longer needed (MADV_DONTNEED).
    ///
    /// Called after all weight tensors have been uploaded to the GPU so that
    /// the kernel can reclaim the file page cache and reduce RssFile.
    /// Only meaningful on Linux; a no-op on other platforms.
    /// Enabled by `LLMRS_MADV_DONTNEED` environment variable.
    #[cfg(target_os = "linux")]
    pub fn madvise_dontneed(&self) {
        let ptr = self.mmap.as_ptr() as *mut libc::c_void;
        let len = self.mmap.len();
        if len == 0 {
            return;
        }
        unsafe {
            let ret = libc::madvise(ptr, len, libc::MADV_DONTNEED);
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                eprintln!("[RSS] madvise(MADV_DONTNEED) failed: {err}");
            } else {
                eprintln!(
                    "[RSS] madvise(MADV_DONTNEED) applied to GGUF mmap ({} MB)",
                    len / (1024 * 1024)
                );
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn madvise_dontneed(&self) {}

    /// Get the raw byte data for a tensor (zero-copy slice into the mmap).
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> &[u8] {
        let start = self.tensor_data_offset + info.offset as usize;
        let size = tensor_byte_size(info);
        &self.mmap[start..start + size]
    }

    /// Full mmap byte slice. Used by `SecondaryMmap` to serve tensor slices
    /// from pre-computed offsets without re-invoking `tensor_data`.
    pub fn mmap_data(&self) -> &[u8] {
        &self.mmap
    }

    /// Absolute offset of the tensor data section within the mmap.
    pub fn tensor_data_offset(&self) -> usize {
        self.tensor_data_offset
    }

    // ---- Metadata accessors ----

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GgufValue::U32(v)) => Some(*v),
            Some(GgufValue::U64(v)) => Some(*v as u32),
            Some(GgufValue::I32(v)) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(GgufValue::U64(v)) => Some(*v),
            Some(GgufValue::U32(v)) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(GgufValue::F32(v)) => Some(*v),
            Some(GgufValue::F64(v)) => Some(*v as f32),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute byte size of a tensor's data based on ggml_type and dims.
pub(crate) fn tensor_byte_size(info: &GgufTensorInfo) -> usize {
    let num_elements: u64 = info.dims.iter().product();
    match info.ggml_type {
        GGML_TYPE_F32 => num_elements as usize * 4,
        GGML_TYPE_F16 => num_elements as usize * 2,
        GGML_TYPE_Q4_0 => {
            // 32 elements per block, 18 bytes per block
            let n_blocks = num_elements as usize / 32;
            n_blocks * 18
        }
        GGML_TYPE_Q4_1 => {
            // 32 elements per block, 20 bytes per block
            let n_blocks = num_elements as usize / 32;
            n_blocks * 20
        }
        GGML_TYPE_Q8_0 => {
            // 32 elements per block, 34 bytes per block
            let n_blocks = num_elements as usize / 32;
            n_blocks * 34
        }
        GGML_TYPE_Q4_K => {
            // 256 elements per super-block, 144 bytes per super-block
            let n_blocks = num_elements as usize / 256;
            n_blocks * 144
        }
        _ => 0, // unsupported types get 0; validated elsewhere
    }
}

/// Check if a ggml_type requires dequantization to F32 at load time.
fn needs_dequant_fallback(ggml_type: u32) -> bool {
    matches!(ggml_type, GGML_TYPE_Q4_K)
}

/// Does this GGUF tensor name refer to the token embedding or the lm_head
/// (untied output projection)? Used to route these tensors through the F16
/// dequant-target path instead of the default F32 so that the ~1 GiB F32
/// GPU upload for 1B-class Llama models (vocab 128256 × hidden 2048) becomes
/// a ~501 MiB F16 upload.
///
/// Names follow llama.cpp convert_hf_to_gguf.py:
///   * `token_embd.weight`  — input embedding table
///   * `output.weight`      — untied lm_head (absent for tied-weight models)
fn is_embed_or_lm_head(name: &str) -> bool {
    name == "token_embd.weight" || name == "output.weight"
}

/// Human-readable name for a ggml_type (for diagnostics).
fn ggml_type_name(ggml_type: u32) -> &'static str {
    match ggml_type {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        GGML_TYPE_Q4_0 => "Q4_0",
        GGML_TYPE_Q4_1 => "Q4_1",
        GGML_TYPE_Q8_0 => "Q8_0",
        GGML_TYPE_Q4_K => "Q4_K",
        10 => "Q2_K",
        11 => "Q3_K",
        13 => "Q5_K",
        14 => "Q6_K",
        _ => "unknown",
    }
}

/// Convert a Vec<f32> into Vec<u8> by reinterpreting the memory (zero-copy).
fn f32_vec_to_u8(v: Vec<f32>) -> Vec<u8> {
    let mut v = std::mem::ManuallyDrop::new(v);
    let ptr = v.as_mut_ptr() as *mut u8;
    let len = v.len() * 4;
    let cap = v.capacity() * 4;
    // Safety: f32 has alignment >= u8, and we correctly scale len/cap by size_of::<f32>().
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

/// Convert ggml_type to our DType.
pub(crate) fn ggml_type_to_dtype(ggml_type: u32) -> Result<DType> {
    match ggml_type {
        GGML_TYPE_F32 => Ok(DType::F32),
        GGML_TYPE_F16 => Ok(DType::F16),
        GGML_TYPE_Q4_0 => Ok(DType::Q4_0),
        GGML_TYPE_Q4_1 => Ok(DType::Q4_1),
        GGML_TYPE_Q8_0 => Ok(DType::Q8_0),
        // K-quants and other types
        4 => Err(anyhow!("GGUF: Q4_1 (old) type 4 is not supported")),
        5 => Err(anyhow!("GGUF: Q4_2 (deprecated) type 5 is not supported")),
        6 => Err(anyhow!(
            "GGUF: Q5_0 type is not supported. Use Q4_0 or Q8_0 quantization."
        )),
        7 => Err(anyhow!(
            "GGUF: Q5_1 type is not supported. Use Q4_0 or Q8_0 quantization."
        )),
        10 => Err(anyhow!(
            "GGUF: Q2_K type is not supported. Use Q4_0 or Q8_0 quantization."
        )),
        11 => Err(anyhow!(
            "GGUF: Q3_K type is not supported. Use Q4_0 or Q8_0 quantization."
        )),
        12 => Err(anyhow!(
            "GGUF: Q4_K type is not supported. Use Q4_0 or Q8_0 quantization."
        )),
        13 => Err(anyhow!(
            "GGUF: Q5_K type is not supported. Use Q4_0 or Q8_0 quantization."
        )),
        14 => Err(anyhow!(
            "GGUF: Q6_K type is not supported. Use Q4_0 or Q8_0 quantization."
        )),
        _ => Err(anyhow!(
            "GGUF: unsupported ggml_type {}. Only F32, F16, Q4_0, Q4_1, and Q8_0 are supported.",
            ggml_type
        )),
    }
}

/// Row-level inverse permutation that undoes llama.cpp's
/// `convert_hf_to_gguf.py` Q/K weight reordering.
///
/// Background — Llama-family models use the NeoX (half-half) RoPE layout in
/// HuggingFace: for a head of size `D`, the first `D/2` elements rotate with
/// the last `D/2`. llama.cpp's ggml originally targeted the "NORM"
/// (interleaved `(x0,x1), (x2,x3), ...`) RoPE. To bridge the two, the GGUF
/// converter permutes each head's rows as:
///
/// ```text
///   weights.reshape(n_head, head_dim/2, 2, *rest).swapaxes(1, 2).reshape(*)
/// ```
///
/// So row `j` (within a head) in the GGUF file holds what HF stored at row
/// `2*j`     if `j <  head_dim/2`
/// `2*j - head_dim + 1` otherwise.
///
/// We use the HF (NeoX) RoPE layout in `kernel_rope_simple`, so we must
/// undo the permute at load time. Every row is `row_size_bytes` wide and
/// rows are contiguous, so this works uniformly for F16/Q4_0/Q8_0 — any
/// dtype whose quantization blocks live fully inside one row (true for all
/// currently-supported ggml_types, whose block_size ∈ {1, 32}).
///
/// Qwen2/Gemma3 GGUF converters do **not** apply this permute (they use the
/// NEOX rope path directly), so this is gated to `ModelArch::Llama`.
///
/// Exposed as `pub(crate)` so the Phase 2 `SwapExecutor` can apply the same
/// permutation when materialising a `TransformerLayer` from the secondary
/// mmap — the loader path and the swap path must stay bit-identical.
pub(crate) fn unpermute_qk_rows(
    src: &[u8],
    n_head: usize,
    head_dim: usize,
    row_size_bytes: usize,
) -> Vec<u8> {
    debug_assert!(head_dim.is_multiple_of(2));
    let half = head_dim / 2;
    let total_rows = n_head * head_dim;
    debug_assert_eq!(src.len(), total_rows * row_size_bytes);

    let mut dst = vec![0u8; src.len()];
    for h in 0..n_head {
        let head_base = h * head_dim;
        for j in 0..head_dim {
            // HF row (h*head_dim + j) <- GGUF row (h*head_dim + src_offset)
            let src_in_head = if j < half { 2 * j } else { 2 * (j - half) + 1 };
            let src_row = head_base + src_in_head;
            let dst_row = head_base + j;
            let src_off = src_row * row_size_bytes;
            let dst_off = dst_row * row_size_bytes;
            dst[dst_off..dst_off + row_size_bytes]
                .copy_from_slice(&src[src_off..src_off + row_size_bytes]);
        }
    }
    dst
}

/// Return `Some((n_head, head_dim))` if the tensor named `name` must be
/// inverse-permuted on load for the given model config — i.e. a Llama-arch
/// `blk.N.attn_q.weight` or `blk.N.attn_k.weight`. Otherwise `None`.
///
/// `pub(crate)` for the same reason as `unpermute_qk_rows` — the swap
/// executor needs the exact same gating to decide whether to permute.
pub(crate) fn qk_permute_shape(name: &str, config: &ModelConfig) -> Option<(usize, usize)> {
    use crate::model_config::ModelArch;
    if config.arch != ModelArch::Llama {
        return None;
    }
    // Match "blk.<N>.attn_q.weight" or "blk.<N>.attn_k.weight".
    let stem = name.strip_prefix("blk.")?;
    let (_idx, rest) = stem.split_once('.')?;
    let head_dim = config.head_dim;
    match rest {
        "attn_q.weight" => Some((config.num_attention_heads, head_dim)),
        "attn_k.weight" => Some((config.num_key_value_heads, head_dim)),
        _ => None,
    }
}

/// Construct a [`ModelConfig`] from GGUF metadata.
///
/// GGUF metadata keys follow the pattern `{arch}.{param_name}` where
/// `{arch}` comes from `general.architecture`.
pub fn parse_model_config(gguf: &GgufFile) -> Result<ModelConfig> {
    use crate::model_config::ModelArch;

    let arch_str = gguf
        .get_str("general.architecture")
        .ok_or_else(|| anyhow!("GGUF: missing 'general.architecture' metadata"))?;

    let arch = match arch_str {
        "llama" => ModelArch::Llama,
        "qwen2" => ModelArch::Qwen2,
        "gemma" | "gemma2" | "gemma3" => ModelArch::Gemma3,
        _ => anyhow::bail!("GGUF: unsupported architecture '{}'", arch_str),
    };

    let prefix = arch_str;

    let hidden_size =
        gguf.get_u32(&format!("{prefix}.embedding_length"))
            .ok_or_else(|| anyhow!("GGUF: missing {prefix}.embedding_length"))? as usize;
    let num_hidden_layers =
        gguf.get_u32(&format!("{prefix}.block_count"))
            .ok_or_else(|| anyhow!("GGUF: missing {prefix}.block_count"))? as usize;
    let num_attention_heads = gguf
        .get_u32(&format!("{prefix}.attention.head_count"))
        .ok_or_else(|| anyhow!("GGUF: missing {prefix}.attention.head_count"))?
        as usize;
    let num_key_value_heads = gguf
        .get_u32(&format!("{prefix}.attention.head_count_kv"))
        .unwrap_or(num_attention_heads as u32) as usize;
    let intermediate_size =
        gguf.get_u32(&format!("{prefix}.feed_forward_length"))
            .ok_or_else(|| anyhow!("GGUF: missing {prefix}.feed_forward_length"))? as usize;
    let vocab_size = gguf
        .get_u32(&format!("{prefix}.vocab_size"))
        .map(|v| v as usize)
        .or_else(|| match gguf.metadata.get("tokenizer.ggml.tokens") {
            Some(GgufValue::Array(arr)) => Some(arr.len()),
            _ => None,
        })
        .unwrap_or(32000);
    let rms_norm_eps = gguf
        .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5) as f64;
    let rope_theta = gguf
        .get_f32(&format!("{prefix}.rope.freq_base"))
        .unwrap_or(10000.0) as f64;
    // head_dim: use explicit GGUF metadata if available (Gemma3 has head_dim != hidden/heads).
    // Gemma3 GGUF stores head_dim as `attention.key_length`, not `attention.head_dim`.
    let head_dim = gguf
        .get_u32(&format!("{prefix}.attention.head_dim"))
        .or_else(|| gguf.get_u32(&format!("{prefix}.attention.key_length")))
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_attention_heads);

    let has_qkv_bias = matches!(arch, ModelArch::Qwen2);
    let tie_word_embeddings = gguf.find_tensor("output.weight").is_none();

    // Gemma3 specific fields
    let (
        rope_local_theta,
        sliding_window,
        sliding_window_pattern,
        query_pre_attn_scalar,
        embed_scale,
    ) = match arch {
        ModelArch::Gemma3 => {
            let local_theta = gguf
                .get_f32(&format!("{prefix}.rope.local.freq_base"))
                .unwrap_or(10000.0) as f64;
            let sw = gguf
                .get_u32(&format!("{prefix}.attention.sliding_window"))
                .map(|v| v as usize);
            let sw_pattern = gguf
                .get_u32(&format!("{prefix}.attention.sliding_window_pattern"))
                .map(|v| v as usize);
            let qpas = gguf
                .get_u32(&format!("{prefix}.attention.query_pre_attn_scalar"))
                .map(|v| v as usize);
            let es = Some((hidden_size as f32).sqrt());
            (Some(local_theta), sw, sw_pattern, qpas, es)
        }
        _ => (None, None, None, None, None),
    };

    // eos_token_id: GGUF may store it as an array or a single value
    let eos_token_id = gguf
        .get_u32("tokenizer.ggml.eos_token_id")
        .unwrap_or(u32::MAX);

    Ok(ModelConfig {
        arch,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        rms_norm_eps,
        rope_theta,
        has_qkv_bias,
        tie_word_embeddings,
        eos_token_id,
        rope_local_theta,
        sliding_window,
        sliding_window_pattern,
        query_pre_attn_scalar,
        embed_scale,
        weight_prefix: String::new(),
    })
}

/// Determine the "weight dtype" of a GGUF file by looking at the dominant
/// ggml_type across weight tensors (excluding norms and biases).
fn detect_weight_dtype(gguf: &GgufFile) -> DType {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for t in &gguf.tensors {
        // Skip norm/bias tensors (they are typically F32)
        if t.name.contains("norm") || t.name.contains("bias") || t.name == "token_embd.weight" {
            continue;
        }
        *counts.entry(t.ggml_type).or_default() += 1;
    }
    // Find the most common type
    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .and_then(|(t, _)| ggml_type_to_dtype(t).ok())
        .unwrap_or(DType::F16)
}

// ---------------------------------------------------------------------------
// GgufSource
// ---------------------------------------------------------------------------

pub struct GgufSource {
    gguf: GgufFile,
    config: ModelConfig,
    weight_dtype: DType,
    cpu_backend: Arc<CpuBackend>,
}

impl GgufSource {
    /// Open a GGUF model file and parse metadata + tensor info.
    pub fn open(path: &Path) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let config = parse_model_config(&gguf)?;
        let weight_dtype = detect_weight_dtype(&gguf);

        eprintln!(
            "[GGUF] Loaded: {} tensors, arch={:?}, weight_dtype={:?}",
            gguf.tensors.len(),
            config.arch,
            weight_dtype
        );

        Ok(Self {
            gguf,
            config,
            weight_dtype,
            cpu_backend: Arc::new(CpuBackend::new()),
        })
    }

    /// Access the underlying parsed GGUF file (metadata, tensor index, mmap).
    /// Used by weight-swap infrastructure to validate a secondary GGUF against
    /// the primary layout without re-parsing.
    pub fn gguf_file(&self) -> &GgufFile {
        &self.gguf
    }

    /// Hint the OS that the GGUF mmap pages are no longer needed.
    /// Forwards to `GgufFile::madvise_dontneed`. See that method for details.
    pub fn madvise_dontneed(&self) {
        self.gguf.madvise_dontneed();
    }

    /// Resolve a TensorId to the GGUF tensor name.
    fn resolve_name(&self, id: &TensorId) -> String {
        match id {
            TensorId::Embed => "token_embd.weight".to_string(),
            TensorId::FinalNorm => "output_norm.weight".to_string(),
            TensorId::LmHead => "output.weight".to_string(),
            TensorId::LayerWeight { layer, kind } => {
                let suffix = match kind {
                    LayerWeightKind::Wq => "attn_q.weight",
                    LayerWeightKind::Wk => "attn_k.weight",
                    LayerWeightKind::Wv => "attn_v.weight",
                    LayerWeightKind::Wo => "attn_output.weight",
                    LayerWeightKind::WGate => "ffn_gate.weight",
                    LayerWeightKind::WUp => "ffn_up.weight",
                    LayerWeightKind::WDown => "ffn_down.weight",
                    LayerWeightKind::AttentionNorm => "attn_norm.weight",
                    LayerWeightKind::FfnNorm => "ffn_norm.weight",
                    LayerWeightKind::PreFfnNorm => "pre_ffn_norm.weight",
                    LayerWeightKind::PostFfnNorm => "post_ffn_norm.weight",
                    LayerWeightKind::QNorm => "attn_q_norm.weight",
                    LayerWeightKind::KNorm => "attn_k_norm.weight",
                };
                format!("blk.{layer}.{suffix}")
            }
            TensorId::LayerBias { layer, kind } => {
                let suffix = match kind {
                    LayerBiasKind::Bq => "attn_q.bias",
                    LayerBiasKind::Bk => "attn_k.bias",
                    LayerBiasKind::Bv => "attn_v.bias",
                };
                format!("blk.{layer}.{suffix}")
            }
        }
    }

    /// Load a tensor from the GGUF file onto the given backend.
    fn load_raw(&self, name: &str, is_weight: bool, backend: &Arc<dyn Backend>) -> Result<Tensor> {
        let info = self
            .gguf
            .find_tensor(name)
            .ok_or_else(|| anyhow!("GGUF: tensor '{}' not found", name))?;

        let data = self.gguf.tensor_data(info);

        // Shape: GGUF stores dims in reverse order (innermost first).
        // Argus uses [rows, cols] convention, i.e. outermost first.
        let shape = Shape::new(info.dims.iter().rev().map(|&d| d as usize).collect());
        let num_elements: usize = info.dims.iter().map(|&d| d as usize).product();

        let is_cpu = backend.name().contains("CPU");

        // Check if this is a type that needs dequantization to F32 at load time
        if needs_dequant_fallback(info.ggml_type) {
            // RSS optimization: for embed/lm_head (token_embd.weight, output.weight)
            // the Q4_K → F32 dequant path produces ~1 GiB for Llama 3.2 1B vocabs.
            // Store as F16 instead (half the size, half-precision round only — the
            // Q4_K super-block dequant still runs at full precision first, so no
            // block-layout-change accuracy loss). Downstream `gather` (get_rows.cl
            // F16 variant) and `matmul_transposed` (matmul_f16) already support F16.
            let target_dtype = if is_embed_or_lm_head(name) {
                DType::F16
            } else {
                DType::F32
            };
            return self.load_with_dequant_as(
                info.ggml_type,
                data,
                num_elements,
                shape,
                is_weight,
                backend,
                target_dtype,
            );
        }

        let dtype = ggml_type_to_dtype(info.ggml_type)?;

        // RSS optimization: for embed/lm_head stored as F32 (llama.cpp's default
        // for `token_embd.weight` on many Q4_0 builds), downcast to F16 at load
        // time to halve the GPU upload. The Q4_K path above already handles the
        // K-quant storage variant; this covers the F32 storage variant we see
        // in the distributed `llama3.2-1b-q4_0.gguf`. Half rounding only, no
        // block-layout change; downstream `gather` (get_rows.cl F16 variant) and
        // `matmul_transposed` (matmul_f16) already support F16.
        if info.ggml_type == GGML_TYPE_F32 && is_embed_or_lm_head(name) {
            let cpu_tensor = self.load_f32_as_f16(name, data, num_elements, shape)?;
            return if is_cpu {
                Ok(cpu_tensor)
            } else if is_weight {
                backend.copy_weight_from(&cpu_tensor)
            } else {
                backend.copy_from(&cpu_tensor)
            };
        }

        // Llama Q/K weights must be inverse-permuted to undo llama.cpp's
        // convert_hf_to_gguf reshape — see `unpermute_qk_rows`. This path
        // allocates a fresh owned buffer; the zero-copy mmap fast path below
        // is only used for non-permuted tensors.
        if let Some((n_head, head_dim)) = qk_permute_shape(name, &self.config) {
            let total_rows: usize = shape.dims()[0];
            debug_assert_eq!(total_rows, n_head * head_dim);
            debug_assert!(data.len().is_multiple_of(total_rows));
            let row_size_bytes = data.len() / total_rows;
            let permuted = unpermute_qk_rows(data, n_head, head_dim, row_size_bytes);
            let owned_buf: Arc<dyn crate::buffer::Buffer> =
                Arc::new(SharedBuffer::from_vec(permuted, dtype));
            let cpu_tensor = Tensor::new(
                shape,
                owned_buf,
                self.cpu_backend.clone() as Arc<dyn Backend>,
            );
            return if is_cpu {
                Ok(cpu_tensor)
            } else if is_weight {
                backend.copy_weight_from(&cpu_tensor)
            } else {
                backend.copy_from(&cpu_tensor)
            };
        }

        // For norm/bias tensors (is_weight=false) that are in F32, use MmapBuffer directly.
        // For weight tensors that are already in their native quantized format, also use MmapBuffer.
        // This gives zero-copy access to the GGUF data.
        let abs_offset = self.gguf.tensor_data_offset + info.offset as usize;

        // Safety: abs_offset + data.len() <= mmap.len() (guaranteed by tensor_data())
        let buffer: Arc<dyn crate::buffer::Buffer> = Arc::new(unsafe {
            MmapBuffer::new(self.gguf.mmap.clone(), abs_offset, data.len(), dtype)
        });

        let cpu_tensor = Tensor::new(shape, buffer, self.cpu_backend.clone() as Arc<dyn Backend>);

        if is_cpu {
            Ok(cpu_tensor)
        } else if is_weight {
            backend.copy_weight_from(&cpu_tensor)
        } else {
            backend.copy_from(&cpu_tensor)
        }
    }

    /// Load a tensor as CPU-only.
    fn load_raw_cpu(&self, name: &str, is_weight: bool) -> Result<Tensor> {
        let info = self
            .gguf
            .find_tensor(name)
            .ok_or_else(|| anyhow!("GGUF: tensor '{}' not found", name))?;

        let data = self.gguf.tensor_data(info);
        let shape = Shape::new(info.dims.iter().rev().map(|&d| d as usize).collect());
        let num_elements: usize = info.dims.iter().map(|&d| d as usize).product();

        // Check if this is a type that needs dequantization to F32 at load time
        if needs_dequant_fallback(info.ggml_type) {
            let cpu_backend_arc: Arc<dyn Backend> = self.cpu_backend.clone() as Arc<dyn Backend>;
            // See `load_raw` for the F16 path rationale (embed/lm_head RSS optimization).
            let target_dtype = if is_embed_or_lm_head(name) {
                DType::F16
            } else {
                DType::F32
            };
            return self.load_with_dequant_as(
                info.ggml_type,
                data,
                num_elements,
                shape,
                is_weight,
                &cpu_backend_arc,
                target_dtype,
            );
        }

        let dtype = ggml_type_to_dtype(info.ggml_type)?;

        // See `load_raw` for the F32 → F16 embed/lm_head downcast rationale.
        if info.ggml_type == GGML_TYPE_F32 && is_embed_or_lm_head(name) {
            return self.load_f32_as_f16(name, data, num_elements, shape);
        }

        // Llama Q/K weights must be inverse-permuted — see load_raw above.
        if let Some((n_head, head_dim)) = qk_permute_shape(name, &self.config) {
            let total_rows: usize = shape.dims()[0];
            debug_assert_eq!(total_rows, n_head * head_dim);
            debug_assert!(data.len().is_multiple_of(total_rows));
            let row_size_bytes = data.len() / total_rows;
            let permuted = unpermute_qk_rows(data, n_head, head_dim, row_size_bytes);
            let owned_buf: Arc<dyn crate::buffer::Buffer> =
                Arc::new(SharedBuffer::from_vec(permuted, dtype));
            return Ok(Tensor::new(
                shape,
                owned_buf,
                self.cpu_backend.clone() as Arc<dyn Backend>,
            ));
        }

        let abs_offset = self.gguf.tensor_data_offset + info.offset as usize;

        // Safety: abs_offset + data.len() <= mmap.len()
        let buffer: Arc<dyn crate::buffer::Buffer> = Arc::new(unsafe {
            MmapBuffer::new(self.gguf.mmap.clone(), abs_offset, data.len(), dtype)
        });

        Ok(Tensor::new(
            shape,
            buffer,
            self.cpu_backend.clone() as Arc<dyn Backend>,
        ))
    }

    /// Downcast an F32-stored embed/lm_head tensor to F16 at load time.
    ///
    /// Scope: this is NOT a general F32 → F16 path. It is gated by
    /// `is_embed_or_lm_head(name)` in both `load_raw` and `load_raw_cpu`.
    /// Other F32 tensors (norms, biases, any hidden-state activations that
    /// would hypothetically be stored in F32) continue to use the zero-copy
    /// `MmapBuffer` path unchanged.
    ///
    /// Rationale: on llama.cpp-generated `llama3.2-1b-q4_0.gguf`,
    /// `token_embd.weight` is stored as F32 (2048 × 128256 ≈ 1.0 GiB). On an
    /// ARM64 Android device the full F32 tensor is uploaded to GPU and the
    /// mmap-backed host copy stays resident until eviction, which adds up to
    /// a ~500 MiB RSS pressure point. Downcasting to F16 halves both the GPU
    /// allocation and the owned host copy. Downstream `gather` (get_rows.cl
    /// F16 variant) and `matmul_transposed` (mul_mv_f16) already support F16.
    ///
    /// Accuracy: single round-to-nearest-even half cast per element. The top-1
    /// argmax on greedy decode matches F32 storage in our measurements; bit-
    /// exactness is not expected because of the rounding.
    fn load_f32_as_f16(
        &self,
        name: &str,
        data: &[u8],
        num_elements: usize,
        shape: Shape,
    ) -> Result<Tensor> {
        use half::f16;

        eprintln!(
            "[GGUF] Downcasting {} tensor F32\u{2192}F16 at load time ({} elements)",
            name, num_elements
        );

        ensure!(
            data.len() == num_elements * 4,
            "GGUF: F32 tensor '{}' has {} bytes but expected {} ({} elements × 4)",
            name,
            data.len(),
            num_elements * 4,
            num_elements
        );

        // F32 mmap bytes → F16 owned bytes. We do not require F32 alignment in
        // the mmap (GGUF aligns tensor data to 32 B by default, but we stay
        // defensive with `from_le_bytes`).
        let mut bytes = Vec::<u8>::with_capacity(num_elements * 2);
        for chunk in data.chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let h = f16::from_f32(v);
            bytes.extend_from_slice(&h.to_bits().to_le_bytes());
        }
        debug_assert_eq!(bytes.len(), num_elements * 2);

        let buffer: Arc<dyn crate::buffer::Buffer> =
            Arc::new(SharedBuffer::from_vec(bytes, DType::F16));

        Ok(Tensor::new(
            shape,
            buffer,
            self.cpu_backend.clone() as Arc<dyn Backend>,
        ))
    }

    /// Dequantize a tensor with an unsupported quant type and store it at the
    /// requested dtype. Supports `DType::F32` (reference path) and
    /// `DType::F16` (half-precision storage, used for embed/lm_head on Q4_0
    /// models whose `token_embd.weight` is actually stored as Q4_K — see
    /// `loader/mod.rs` embed/lm_head load sites).
    ///
    /// F16 path:
    /// 1. Dequant to `Vec<f32>` (unchanged cold path).
    /// 2. Cast each element via `half::f16::from_f32` (round-to-nearest-even).
    /// 3. Publish as `SharedBuffer` with `DType::F16`.
    ///
    /// Downstream `backend.copy_weight_from` allocates at F16 size (2 bytes
    /// per element, half of the F32 path) — for Llama 3.2 1B embed this
    /// drops the single ~1 GiB F32 GPU upload to ~501 MiB.
    ///
    /// Accuracy: half rounding only; the Q4_K super-block structure of the
    /// source is preserved through the full-precision dequant step. No
    /// re-quantization into a different block layout.
    #[allow(clippy::too_many_arguments)]
    fn load_with_dequant_as(
        &self,
        ggml_type: u32,
        data: &[u8],
        num_elements: usize,
        shape: Shape,
        is_weight: bool,
        backend: &Arc<dyn Backend>,
        target_dtype: DType,
    ) -> Result<Tensor> {
        let type_name = ggml_type_name(ggml_type);
        eprintln!(
            "[GGUF] Dequantizing {} tensor ({} elements) to {:?} at load time",
            type_name, num_elements, target_dtype
        );

        let f32_data = match ggml_type {
            GGML_TYPE_Q4_K => dequant_q4_k(data, num_elements),
            GGML_TYPE_Q4_1 => dequant_q4_1(data, num_elements),
            _ => {
                return Err(anyhow!(
                    "GGUF: no dequant implementation for type {} ({})",
                    type_name,
                    ggml_type
                ));
            }
        };

        let buffer: Arc<dyn crate::buffer::Buffer> = match target_dtype {
            DType::F32 => {
                // Zero-copy reinterpret the Vec<f32> as Vec<u8>.
                let byte_data = f32_vec_to_u8(f32_data);
                Arc::new(SharedBuffer::from_vec(byte_data, DType::F32))
            }
            DType::F16 => {
                use half::f16;
                // SAFETY: we own `f32_data` and consume it into a new Vec<u8>.
                // Round-to-nearest-even half rounding is applied element-wise.
                let mut bytes = Vec::<u8>::with_capacity(num_elements * 2);
                for v in f32_data.into_iter() {
                    let h = f16::from_f32(v);
                    bytes.extend_from_slice(&h.to_bits().to_le_bytes());
                }
                Arc::new(SharedBuffer::from_vec(bytes, DType::F16))
            }
            other => {
                return Err(anyhow!(
                    "GGUF: load_with_dequant_as unsupported target dtype {:?}",
                    other
                ));
            }
        };

        let cpu_tensor = Tensor::new(shape, buffer, self.cpu_backend.clone() as Arc<dyn Backend>);

        let is_cpu = backend.name().contains("CPU");
        if is_cpu {
            Ok(cpu_tensor)
        } else if is_weight {
            backend.copy_weight_from(&cpu_tensor)
        } else {
            backend.copy_from(&cpu_tensor)
        }
    }

    /// Check if a tensor exists in the GGUF file.
    fn has_raw(&self, name: &str) -> bool {
        self.gguf.find_tensor(name).is_some()
    }
}

// ---------------------------------------------------------------------------
// TensorSource implementation
// ---------------------------------------------------------------------------

impl TensorSource for GgufSource {
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
        self.load_raw(&name, is_weight, backend)
    }

    fn load_tensor_cpu(
        &self,
        id: &TensorId,
        is_weight: bool,
        _memory: &dyn Memory,
    ) -> Result<Tensor> {
        let name = self.resolve_name(id);
        self.load_raw_cpu(&name, is_weight)
    }

    fn has_tensor(&self, id: &TensorId) -> bool {
        let name = self.resolve_name(id);
        self.has_raw(&name)
    }

    fn cpu_backend(&self) -> Arc<dyn Backend> {
        self.cpu_backend.clone() as Arc<dyn Backend>
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Q/K unpermute helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_unpermute_qk_rows_matches_numpy_reshape_swapaxes() {
        // Build an n_head=2, head_dim=4, k=1 (row_size=4 bytes) tensor whose
        // rows are byte-tagged so we can confirm the destination layout.
        //
        // "GGUF-stored" representation (input to unpermute):
        //   HF would store row `(h * head_dim + 2*p + s)` with value tagged
        //   (h, 2*p + s). llama.cpp's convert permute reshuffles so that the
        //   saved row at position `(h * head_dim + s * half + p)` holds the
        //   data that HF had at `(h * head_dim + 2*p + s)`.
        //
        // So if we fill the input with that saved-order tag, `unpermute_qk_rows`
        // should put the HF-order tag back: row `(h*head_dim + j)` maps to
        // `(h, j)` where j = HF within-head index.
        let n_head = 2;
        let head_dim = 4;
        let half = head_dim / 2;
        let row_size_bytes = 1;
        let mut input = vec![0u8; n_head * head_dim * row_size_bytes];
        for h in 0..n_head {
            for s in 0..2 {
                for p in 0..half {
                    // GGUF saved order: within-head index = s*half + p.
                    let saved_idx = s * half + p;
                    // HF original within-head index = 2*p + s.
                    let hf_idx = 2 * p + s;
                    // Tag byte = head * 16 + HF within-head index (0..15).
                    input[h * head_dim + saved_idx] = ((h * head_dim + hf_idx) & 0xff) as u8;
                }
            }
        }

        let out = unpermute_qk_rows(&input, n_head, head_dim, row_size_bytes);
        // After unpermute, row `h*head_dim + j` should hold tag `h*head_dim + j`.
        for h in 0..n_head {
            for j in 0..head_dim {
                assert_eq!(
                    out[h * head_dim + j],
                    ((h * head_dim + j) & 0xff) as u8,
                    "h={h}, j={j}"
                );
            }
        }
    }

    #[test]
    fn test_unpermute_qk_rows_wide_rows() {
        // Same n_head/head_dim but a 5-byte row (exercises per-row memcpy).
        let n_head = 3;
        let head_dim = 4;
        let half = head_dim / 2;
        let row_size = 5;
        let mut input = vec![0u8; n_head * head_dim * row_size];
        for h in 0..n_head {
            for s in 0..2 {
                for p in 0..half {
                    let saved_idx = s * half + p;
                    let hf_idx = 2 * p + s;
                    let tag = (h * head_dim + hf_idx) as u8;
                    let off = (h * head_dim + saved_idx) * row_size;
                    for b in 0..row_size {
                        input[off + b] = tag.wrapping_add(b as u8);
                    }
                }
            }
        }
        let out = unpermute_qk_rows(&input, n_head, head_dim, row_size);
        for h in 0..n_head {
            for j in 0..head_dim {
                let tag = (h * head_dim + j) as u8;
                let off = (h * head_dim + j) * row_size;
                for b in 0..row_size {
                    assert_eq!(
                        out[off + b],
                        tag.wrapping_add(b as u8),
                        "h={h}, j={j}, b={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_qk_permute_shape_gating() {
        use crate::model_config::ModelArch;
        fn make_cfg(arch: ModelArch) -> ModelConfig {
            ModelConfig {
                arch,
                hidden_size: 2048,
                num_hidden_layers: 16,
                num_attention_heads: 32,
                num_key_value_heads: 8,
                head_dim: 64,
                intermediate_size: 8192,
                vocab_size: 128256,
                rms_norm_eps: 1e-5,
                rope_theta: 500000.0,
                has_qkv_bias: false,
                tie_word_embeddings: true,
                eos_token_id: 128009,
                rope_local_theta: None,
                sliding_window: None,
                sliding_window_pattern: None,
                query_pre_attn_scalar: None,
                embed_scale: None,
                weight_prefix: String::new(),
            }
        }
        let base = make_cfg(ModelArch::Llama);
        // Llama Q/K: return shape.
        assert_eq!(
            qk_permute_shape("blk.0.attn_q.weight", &base),
            Some((32, 64))
        );
        assert_eq!(
            qk_permute_shape("blk.7.attn_k.weight", &base),
            Some((8, 64))
        );
        // Llama V / output / norms: skip.
        assert_eq!(qk_permute_shape("blk.0.attn_v.weight", &base), None);
        assert_eq!(qk_permute_shape("blk.0.attn_output.weight", &base), None);
        assert_eq!(qk_permute_shape("blk.0.attn_norm.weight", &base), None);
        assert_eq!(qk_permute_shape("token_embd.weight", &base), None);

        // Qwen2: skip (converter does not permute).
        let qwen = make_cfg(ModelArch::Qwen2);
        assert_eq!(qk_permute_shape("blk.0.attn_q.weight", &qwen), None);
        assert_eq!(qk_permute_shape("blk.0.attn_k.weight", &qwen), None);
    }

    // -----------------------------------------------------------------------
    // GgufTestBuilder: synthesize valid GGUF v3 binaries in memory
    // -----------------------------------------------------------------------

    struct GgufTestBuilder {
        metadata: Vec<(String, u32, Vec<u8>)>, // (key, type_tag, encoded_value)
        tensors: Vec<TestTensor>,
        alignment: usize,
    }

    struct TestTensor {
        name: String,
        dims: Vec<u64>,
        ggml_type: u32,
        data: Vec<u8>,
    }

    impl GgufTestBuilder {
        fn new() -> Self {
            Self {
                metadata: Vec::new(),
                tensors: Vec::new(),
                alignment: GGUF_DEFAULT_ALIGNMENT,
            }
        }

        fn add_metadata_str(&mut self, key: &str, val: &str) -> &mut Self {
            let mut buf = Vec::new();
            buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
            buf.extend_from_slice(val.as_bytes());
            self.metadata.push((key.to_string(), GGUF_TYPE_STRING, buf));
            self
        }

        fn add_metadata_u32(&mut self, key: &str, val: u32) -> &mut Self {
            self.metadata
                .push((key.to_string(), GGUF_TYPE_U32, val.to_le_bytes().to_vec()));
            self
        }

        fn add_metadata_f32(&mut self, key: &str, val: f32) -> &mut Self {
            self.metadata
                .push((key.to_string(), GGUF_TYPE_F32, val.to_le_bytes().to_vec()));
            self
        }

        fn add_metadata_u64(&mut self, key: &str, val: u64) -> &mut Self {
            self.metadata
                .push((key.to_string(), GGUF_TYPE_U64, val.to_le_bytes().to_vec()));
            self
        }

        fn add_tensor(
            &mut self,
            name: &str,
            dims: &[u64],
            ggml_type: u32,
            data: &[u8],
        ) -> &mut Self {
            self.tensors.push(TestTensor {
                name: name.to_string(),
                dims: dims.to_vec(),
                ggml_type,
                data: data.to_vec(),
            });
            self
        }

        fn build(&self) -> Vec<u8> {
            let mut buf = Vec::new();

            // Header
            buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
            buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            buf.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

            // KV metadata
            for (key, type_tag, value_bytes) in &self.metadata {
                // GGUF string: u64 len + bytes
                buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
                buf.extend_from_slice(key.as_bytes());
                buf.extend_from_slice(&type_tag.to_le_bytes());
                buf.extend_from_slice(value_bytes);
            }

            // Tensor infos (compute offsets sequentially with alignment)
            let mut data_offset: u64 = 0;
            let mut tensor_offsets = Vec::with_capacity(self.tensors.len());
            for t in &self.tensors {
                // Align data_offset
                let rem = data_offset as usize % self.alignment;
                if rem != 0 {
                    data_offset += (self.alignment - rem) as u64;
                }
                tensor_offsets.push(data_offset);
                data_offset += t.data.len() as u64;
            }

            for (i, t) in self.tensors.iter().enumerate() {
                // name
                buf.extend_from_slice(&(t.name.len() as u64).to_le_bytes());
                buf.extend_from_slice(t.name.as_bytes());
                // n_dims
                buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
                // dims
                for &d in &t.dims {
                    buf.extend_from_slice(&d.to_le_bytes());
                }
                // ggml_type
                buf.extend_from_slice(&t.ggml_type.to_le_bytes());
                // offset
                buf.extend_from_slice(&tensor_offsets[i].to_le_bytes());
            }

            // Pad to alignment before tensor data
            let rem = buf.len() % self.alignment;
            if rem != 0 {
                buf.resize(buf.len() + (self.alignment - rem), 0);
            }

            // Tensor data (with alignment padding between tensors)
            #[allow(clippy::never_loop)]
            for (i, _t) in self.tensors.iter().enumerate() {
                // Pad to reach the expected offset
                let expected_pos = tensor_offsets[i] as usize;
                let current_data_pos = buf.len()
                    - (buf.len()
                        - self
                            .tensors
                            .first()
                            .map(|_| {
                                // Find where tensor data section starts
                                // It starts right after the alignment-padded header+info
                                // We need to know the data section start
                                0 // placeholder, we handle below
                            })
                            .unwrap_or(0));
                // Actually, since we've already aligned and are now writing data
                // sequentially, we just need to pad between tensors.
                let data_section_start = buf.len()
                    - self.tensors[..i]
                        .iter()
                        .enumerate()
                        .map(|(j, tt)| {
                            let pad = if j == 0 {
                                tensor_offsets[0] as usize
                            } else {
                                (tensor_offsets[j] - tensor_offsets[j - 1]) as usize
                                    - self.tensors[j - 1].data.len()
                            };
                            tt.data.len() + pad
                        })
                        .sum::<usize>();

                // Simpler approach: just ensure buf grows to data_section_start + offset
                // But we don't know data_section_start... Let's track it differently.
                // Actually let's just re-do this more simply.

                let _ = (current_data_pos, expected_pos, data_section_start); // suppress warnings
                break; // We'll rewrite below
            }

            // Rewrite tensor data section more simply
            // First, remove whatever we added in the loop above
            // Actually the loop above broke immediately, so buf is still just
            // header+metadata+tensor_infos+alignment_padding.
            let data_section_start = buf.len();

            for (i, t) in self.tensors.iter().enumerate() {
                let target_len = data_section_start + tensor_offsets[i] as usize;
                if buf.len() < target_len {
                    buf.resize(target_len, 0);
                }
                buf.extend_from_slice(&t.data);
            }

            buf
        }
    }

    // Helper: create a simple GGUF with one F32 tensor for basic parsing tests
    fn make_simple_gguf() -> Vec<u8> {
        let tensor_data: Vec<u8> = (0..16u32).flat_map(|v| v.to_le_bytes().to_vec()).collect(); // 16 f32 values = 64 bytes
        let mut builder = GgufTestBuilder::new();
        builder
            .add_metadata_str("general.architecture", "llama")
            .add_metadata_u32("llama.embedding_length", 64)
            .add_metadata_u32("llama.block_count", 2)
            .add_metadata_u32("llama.attention.head_count", 4)
            .add_metadata_u32("llama.attention.head_count_kv", 2)
            .add_metadata_u32("llama.feed_forward_length", 128)
            .add_metadata_u32("llama.vocab_size", 256)
            .add_metadata_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .add_metadata_f32("llama.rope.freq_base", 10000.0)
            .add_metadata_u32("llama.context_length", 2048)
            .add_tensor("test.weight", &[16, 1], GGML_TYPE_F32, &tensor_data);
        builder.build()
    }

    fn parse_from_bytes(data: &[u8]) -> Result<GgufFile> {
        // Write to a temp file and mmap it
        let tmp = std::env::temp_dir().join(format!(
            "argus_engine_gguf_test_{}.gguf",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&tmp, data)?;
        let result = GgufFile::open(&tmp);
        let _ = std::fs::remove_file(&tmp);
        result
    }

    // -----------------------------------------------------------------------
    // Test cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_gguf_parse_header() {
        let data = make_simple_gguf();
        let gguf = parse_from_bytes(&data).expect("Failed to parse GGUF");
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.metadata.len(), 10);
    }

    #[test]
    fn test_gguf_parse_metadata() {
        let data = make_simple_gguf();
        let gguf = parse_from_bytes(&data).expect("Failed to parse GGUF");

        assert_eq!(gguf.get_str("general.architecture"), Some("llama"));
        assert_eq!(gguf.get_u32("llama.embedding_length"), Some(64));
        assert_eq!(gguf.get_u32("llama.block_count"), Some(2));
        assert_eq!(gguf.get_f32("llama.rope.freq_base"), Some(10000.0));
    }

    #[test]
    fn test_gguf_parse_tensor_info() {
        let data = make_simple_gguf();
        let gguf = parse_from_bytes(&data).expect("Failed to parse GGUF");

        let info = gguf.find_tensor("test.weight").expect("Tensor not found");
        assert_eq!(info.name, "test.weight");
        assert_eq!(info.dims, vec![16, 1]);
        assert_eq!(info.ggml_type, GGML_TYPE_F32);
    }

    #[test]
    fn test_gguf_invalid_magic() {
        let mut data = make_simple_gguf();
        // Corrupt the magic bytes
        data[0] = 0xFF;
        data[1] = 0xFF;
        let result = parse_from_bytes(&data);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("invalid magic"),
            "Expected 'invalid magic' error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_gguf_unsupported_version() {
        let mut data = make_simple_gguf();
        // Change version to 2
        let v2_bytes = 2u32.to_le_bytes();
        data[4] = v2_bytes[0];
        data[5] = v2_bytes[1];
        data[6] = v2_bytes[2];
        data[7] = v2_bytes[3];
        let result = parse_from_bytes(&data);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("unsupported version"),
            "Expected 'unsupported version' error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_gguf_dim_order() {
        // GGUF dims [4, 3] (inner=4, outer=3) should become Argus shape [3, 4]
        let tensor_data = vec![0u8; 12 * 4]; // 12 f32 values
        let mut builder = GgufTestBuilder::new();
        builder
            .add_metadata_str("general.architecture", "llama")
            .add_tensor("matrix.weight", &[4, 3], GGML_TYPE_F32, &tensor_data);
        let bytes = builder.build();
        let gguf = parse_from_bytes(&bytes).expect("Failed to parse GGUF");
        let info = gguf.find_tensor("matrix.weight").unwrap();

        // GGUF raw dims should be [4, 3]
        assert_eq!(info.dims, vec![4, 3]);

        // When converting to Argus shape: reverse to [3, 4]
        let shape = Shape::new(info.dims.iter().rev().map(|&d| d as usize).collect());
        assert_eq!(shape.dims(), &[3, 4]);
    }

    #[test]
    fn test_gguf_zero_copy_data() {
        // Create a tensor with known data pattern
        let tensor_data: Vec<u8> = (0..8u32)
            .flat_map(|v| (v * 100).to_le_bytes().to_vec())
            .collect(); // 8 f32 values
        let mut builder = GgufTestBuilder::new();
        builder
            .add_metadata_str("general.architecture", "llama")
            .add_tensor("values", &[8], GGML_TYPE_F32, &tensor_data);
        let bytes = builder.build();
        let gguf = parse_from_bytes(&bytes).expect("Failed to parse GGUF");
        let info = gguf.find_tensor("values").unwrap();
        let data = gguf.tensor_data(info);

        // Verify the data matches the original pattern
        assert_eq!(data.len(), 32); // 8 * 4 bytes
        for i in 0..8u32 {
            let expected = (i * 100).to_le_bytes();
            let offset = i as usize * 4;
            assert_eq!(
                &data[offset..offset + 4],
                &expected,
                "Mismatch at element {}",
                i
            );
        }
    }

    #[test]
    fn test_config_from_gguf_llama() {
        let data = make_simple_gguf();
        let gguf = parse_from_bytes(&data).expect("Failed to parse GGUF");
        let config = parse_model_config(&gguf).expect("Failed to create config");

        assert_eq!(config.arch, crate::model_config::ModelArch::Llama);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_hidden_layers, 2);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.intermediate_size, 128);
        assert_eq!(config.vocab_size, 256);
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!((config.rope_theta - 10000.0).abs() < 1.0);
        assert_eq!(config.head_dim, 16); // 64 / 4
        assert!(!config.has_qkv_bias);
    }

    #[test]
    fn test_config_from_gguf_qwen2() {
        let mut builder = GgufTestBuilder::new();
        builder
            .add_metadata_str("general.architecture", "qwen2")
            .add_metadata_u32("qwen2.embedding_length", 1536)
            .add_metadata_u32("qwen2.block_count", 28)
            .add_metadata_u32("qwen2.attention.head_count", 12)
            .add_metadata_u32("qwen2.attention.head_count_kv", 2)
            .add_metadata_u32("qwen2.feed_forward_length", 8960)
            .add_metadata_u32("qwen2.vocab_size", 151936)
            .add_metadata_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
            .add_metadata_f32("qwen2.rope.freq_base", 1000000.0)
            .add_metadata_u32("qwen2.context_length", 32768);
        let bytes = builder.build();
        let gguf = parse_from_bytes(&bytes).expect("Failed to parse GGUF");
        let config = parse_model_config(&gguf).expect("Failed to create config");

        assert_eq!(config.arch, crate::model_config::ModelArch::Qwen2);
        assert!(config.has_qkv_bias);
        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.head_dim, 128); // 1536 / 12
    }

    #[test]
    fn test_ggml_type_to_dtype_supported() {
        assert_eq!(ggml_type_to_dtype(0).unwrap(), DType::F32);
        assert_eq!(ggml_type_to_dtype(1).unwrap(), DType::F16);
        assert_eq!(ggml_type_to_dtype(2).unwrap(), DType::Q4_0);
        assert_eq!(ggml_type_to_dtype(3).unwrap(), DType::Q4_1);
        assert_eq!(ggml_type_to_dtype(8).unwrap(), DType::Q8_0);
    }

    #[test]
    fn test_ggml_type_to_dtype_unsupported() {
        // K-quants should produce clear error messages
        assert!(ggml_type_to_dtype(12).is_err()); // Q4_K
        assert!(ggml_type_to_dtype(13).is_err()); // Q5_K
        assert!(ggml_type_to_dtype(14).is_err()); // Q6_K

        let err = ggml_type_to_dtype(12).unwrap_err().to_string();
        assert!(
            err.contains("Q4_K"),
            "Error should mention Q4_K, got: {}",
            err
        );
    }

    #[test]
    fn test_tensor_byte_size() {
        let make_info = |ggml_type: u32, dims: Vec<u64>| GgufTensorInfo {
            name: "t".to_string(),
            dims,
            ggml_type,
            offset: 0,
        };

        // F32: 16 elements * 4 bytes
        assert_eq!(tensor_byte_size(&make_info(GGML_TYPE_F32, vec![16])), 64);
        // F16: 16 elements * 2 bytes
        assert_eq!(tensor_byte_size(&make_info(GGML_TYPE_F16, vec![16])), 32);
        // Q4_0: 32 elements = 1 block = 18 bytes
        assert_eq!(tensor_byte_size(&make_info(GGML_TYPE_Q4_0, vec![32])), 18);
        // Q8_0: 32 elements = 1 block = 34 bytes
        assert_eq!(tensor_byte_size(&make_info(GGML_TYPE_Q8_0, vec![32])), 34);
        // Q4_0: 2D tensor [32, 4] = 128 elements = 4 blocks = 72 bytes
        assert_eq!(
            tensor_byte_size(&make_info(GGML_TYPE_Q4_0, vec![32, 4])),
            72
        );
        // Q4_K: 256 elements = 1 super-block = 144 bytes
        assert_eq!(tensor_byte_size(&make_info(GGML_TYPE_Q4_K, vec![256])), 144);
        // Q4_K: 2D [256, 4] = 1024 elements = 4 super-blocks = 576 bytes
        assert_eq!(
            tensor_byte_size(&make_info(GGML_TYPE_Q4_K, vec![256, 4])),
            576
        );
    }

    #[test]
    fn test_resolve_name_embed() {
        let data = make_simple_gguf();
        let gguf = parse_from_bytes(&data).expect("Failed to parse GGUF");
        let config = parse_model_config(&gguf).unwrap();
        let source = GgufSource {
            gguf,
            config,
            weight_dtype: DType::Q4_0,
            cpu_backend: Arc::new(CpuBackend::new()),
        };

        assert_eq!(source.resolve_name(&TensorId::Embed), "token_embd.weight");
        assert_eq!(
            source.resolve_name(&TensorId::FinalNorm),
            "output_norm.weight"
        );
        assert_eq!(source.resolve_name(&TensorId::LmHead), "output.weight");
        assert_eq!(
            source.resolve_name(&TensorId::LayerWeight {
                layer: 0,
                kind: LayerWeightKind::Wq
            }),
            "blk.0.attn_q.weight"
        );
        assert_eq!(
            source.resolve_name(&TensorId::LayerWeight {
                layer: 5,
                kind: LayerWeightKind::WDown
            }),
            "blk.5.ffn_down.weight"
        );
        assert_eq!(
            source.resolve_name(&TensorId::LayerBias {
                layer: 1,
                kind: LayerBiasKind::Bq
            }),
            "blk.1.attn_q.bias"
        );
    }

    #[test]
    fn test_detect_weight_dtype() {
        let mut builder = GgufTestBuilder::new();
        builder.add_metadata_str("general.architecture", "llama");
        // Add some Q4_0 weight tensors and F32 norm tensors
        let q4_data = vec![0u8; 18]; // 1 Q4_0 block = 18 bytes
        let f32_data = vec![0u8; 128]; // 32 f32 values
        builder
            .add_tensor("blk.0.attn_q.weight", &[32], GGML_TYPE_Q4_0, &q4_data)
            .add_tensor("blk.0.attn_k.weight", &[32], GGML_TYPE_Q4_0, &q4_data)
            .add_tensor("blk.0.attn_norm.weight", &[32], GGML_TYPE_F32, &f32_data);
        let bytes = builder.build();
        let gguf = parse_from_bytes(&bytes).expect("parse");

        // Q4_0 should win (2 vs 0 after norm exclusion)
        assert_eq!(detect_weight_dtype(&gguf), DType::Q4_0);
    }

    #[test]
    fn test_gguf_multiple_tensors() {
        let mut builder = GgufTestBuilder::new();
        builder.add_metadata_str("general.architecture", "llama");

        let data_a: Vec<u8> = (0..4u32).flat_map(|v| v.to_le_bytes().to_vec()).collect();
        let data_b: Vec<u8> = (100..108u32)
            .flat_map(|v| v.to_le_bytes().to_vec())
            .collect();

        builder
            .add_tensor("a", &[4], GGML_TYPE_F32, &data_a)
            .add_tensor("b", &[8], GGML_TYPE_F32, &data_b);

        let bytes = builder.build();
        let gguf = parse_from_bytes(&bytes).expect("parse");

        assert_eq!(gguf.tensors.len(), 2);

        let info_a = gguf.find_tensor("a").unwrap();
        let slice_a = gguf.tensor_data(info_a);
        assert_eq!(slice_a.len(), 16);
        // First element should be 0
        assert_eq!(u32::from_le_bytes(slice_a[0..4].try_into().unwrap()), 0);

        let info_b = gguf.find_tensor("b").unwrap();
        let slice_b = gguf.tensor_data(info_b);
        assert_eq!(slice_b.len(), 32);
        // First element should be 100
        assert_eq!(u32::from_le_bytes(slice_b[0..4].try_into().unwrap()), 100);
    }

    #[test]
    fn test_gguf_metadata_u64() {
        let mut builder = GgufTestBuilder::new();
        builder
            .add_metadata_str("general.architecture", "llama")
            .add_metadata_u64("some.big_value", 0xDEAD_BEEF_1234_5678);
        let bytes = builder.build();
        let gguf = parse_from_bytes(&bytes).expect("parse");
        assert_eq!(gguf.get_u64("some.big_value"), Some(0xDEAD_BEEF_1234_5678));
    }

    #[test]
    fn test_gguf_q4_k_dequant_load() {
        use half::f16;

        // Build a Q4_K tensor (1 super-block = 256 elements, 144 bytes)
        // d=1.0, dmin=0.0, sub-blocks 0..3 sc=1, all nibbles=5
        let mut q4k_data = [0u8; 144];
        let d_bits = f16::from_f32(1.0).to_bits().to_le_bytes();
        q4k_data[0] = d_bits[0];
        q4k_data[1] = d_bits[1];
        // dmin=0.0 (already zeroed)
        // scales_raw[0..4] = 1
        for i in 0..4 {
            q4k_data[4 + i] = 1;
        }
        // All qs nibbles = 5 => byte = 0x55
        for i in 0..128 {
            q4k_data[16 + i] = 0x55;
        }

        let mut builder = GgufTestBuilder::new();
        builder
            .add_metadata_str("general.architecture", "llama")
            .add_metadata_u32("llama.embedding_length", 256)
            .add_metadata_u32("llama.block_count", 1)
            .add_metadata_u32("llama.attention.head_count", 4)
            .add_metadata_u32("llama.attention.head_count_kv", 2)
            .add_metadata_u32("llama.feed_forward_length", 512)
            .add_metadata_u32("llama.vocab_size", 256)
            .add_metadata_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .add_metadata_f32("llama.rope.freq_base", 10000.0)
            .add_metadata_u32("llama.context_length", 2048)
            .add_tensor(
                "token_embd.weight",
                &[256], // 256 elements in GGUF dim order
                GGML_TYPE_Q4_K,
                &q4k_data,
            );
        let bytes = builder.build();
        let gguf = parse_from_bytes(&bytes).expect("parse");

        let config = parse_model_config(&gguf).unwrap();
        let source = GgufSource {
            gguf,
            config,
            weight_dtype: DType::Q4_0,
            cpu_backend: Arc::new(CpuBackend::new()),
        };

        // Load through load_raw_cpu (the normal embed path).
        // Since 2026-04-24: embed/lm_head tensors that require dequant fallback
        // are stored as F16 instead of F32 to halve the GPU upload size.
        let tensor = source
            .load_raw_cpu("token_embd.weight", false)
            .expect("Q4_K tensor should load via dequant fallback");

        // Verify it was dequantized to F16 (embed/lm_head RSS optimization).
        assert_eq!(tensor.buffer().dtype(), DType::F16);
        assert_eq!(tensor.shape().numel(), 256);

        // Verify values: sub-blocks 0..3 => 1.0 * 1 * 5 = 5.0
        // Half-precision rounding preserves small integers exactly.
        let ptr = tensor.buffer().as_ptr() as *const u16;
        for i in 0..128 {
            let bits = unsafe { *ptr.add(i) };
            let v = f16::from_bits(bits).to_f32();
            assert!(
                (v - 5.0).abs() < 0.01,
                "element {} = {}, expected 5.0",
                i,
                v
            );
        }
        // sub-blocks 4..7 => sc=0, so 0.0
        for i in 128..256 {
            let bits = unsafe { *ptr.add(i) };
            let v = f16::from_bits(bits).to_f32();
            assert!(
                (v - 0.0).abs() < 0.01,
                "element {} = {}, expected 0.0",
                i,
                v
            );
        }
    }

    #[test]
    fn test_gguf_q4_k_dequant_non_embed_stays_f32() {
        // Regression: the F16 storage path must only apply to embed/lm_head.
        // All other Q4_K-stored tensors (should one appear in a hybrid GGUF)
        // continue to use the F32 reference path so unrelated matmul/norm
        // code paths are not affected by this optimization.
        use half::f16;

        // Same Q4_K super-block payload as `test_gguf_q4_k_dequant_load`.
        let mut q4k_data = [0u8; 144];
        let d_bits = f16::from_f32(1.0).to_bits().to_le_bytes();
        q4k_data[0] = d_bits[0];
        q4k_data[1] = d_bits[1];
        for i in 0..4 {
            q4k_data[4 + i] = 1;
        }
        for i in 0..128 {
            q4k_data[16 + i] = 0x55;
        }

        let mut builder = GgufTestBuilder::new();
        builder
            .add_metadata_str("general.architecture", "llama")
            .add_metadata_u32("llama.embedding_length", 256)
            .add_metadata_u32("llama.block_count", 1)
            .add_metadata_u32("llama.attention.head_count", 4)
            .add_metadata_u32("llama.attention.head_count_kv", 2)
            .add_metadata_u32("llama.feed_forward_length", 512)
            .add_metadata_u32("llama.vocab_size", 256)
            .add_metadata_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .add_metadata_f32("llama.rope.freq_base", 10000.0)
            .add_metadata_u32("llama.context_length", 2048)
            // Use a non-embed tensor name: synthetic "some.other.weight".
            .add_tensor("some.other.weight", &[256], GGML_TYPE_Q4_K, &q4k_data);
        let bytes = builder.build();
        let gguf = parse_from_bytes(&bytes).expect("parse");

        let config = parse_model_config(&gguf).unwrap();
        let source = GgufSource {
            gguf,
            config,
            weight_dtype: DType::Q4_0,
            cpu_backend: Arc::new(CpuBackend::new()),
        };

        let tensor = source
            .load_raw_cpu("some.other.weight", false)
            .expect("Q4_K tensor should load via dequant fallback");

        // Non-embed Q4_K → F32 reference path (unchanged).
        assert_eq!(tensor.buffer().dtype(), DType::F32);
        assert_eq!(tensor.shape().numel(), 256);
    }

    #[test]
    fn test_is_embed_or_lm_head() {
        assert!(is_embed_or_lm_head("token_embd.weight"));
        assert!(is_embed_or_lm_head("output.weight"));
        assert!(!is_embed_or_lm_head("blk.0.attn_q.weight"));
        assert!(!is_embed_or_lm_head("output_norm.weight"));
        assert!(!is_embed_or_lm_head("token_embd.bias"));
    }

    #[test]
    fn test_gguf_needs_dequant_fallback() {
        assert!(!needs_dequant_fallback(GGML_TYPE_F32));
        assert!(!needs_dequant_fallback(GGML_TYPE_F16));
        assert!(!needs_dequant_fallback(GGML_TYPE_Q4_0));
        assert!(!needs_dequant_fallback(GGML_TYPE_Q4_1));
        assert!(!needs_dequant_fallback(GGML_TYPE_Q8_0));
        assert!(needs_dequant_fallback(GGML_TYPE_Q4_K));
    }

    /// F32-stored `token_embd.weight` must be downcast to F16 at load time.
    /// This is the actual storage format of llama.cpp-generated
    /// `llama3.2-1b-q4_0.gguf` (confirmed via `gguf-dump`), so the Q4_K
    /// dequant path alone does not achieve the RSS savings from commit
    /// 11af06d. Regression guard for the F32 → F16 downcast added
    /// 2026-04-24.
    #[test]
    fn test_gguf_f32_embed_downcasts_to_f16() {
        use half::f16;

        // 256 F32 values = 1024 bytes. Use a mix including non-integer
        // reals to exercise the half rounding path, plus values that
        // survive half rounding exactly.
        let mut f32_bytes = Vec::<u8>::with_capacity(256 * 4);
        for i in 0..256 {
            // Deterministic pattern: 0.0, 0.5, 1.0, 1.5, ...
            let v: f32 = (i as f32) * 0.5;
            f32_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let mut builder = GgufTestBuilder::new();
        builder
            .add_metadata_str("general.architecture", "llama")
            .add_metadata_u32("llama.embedding_length", 256)
            .add_metadata_u32("llama.block_count", 1)
            .add_metadata_u32("llama.attention.head_count", 4)
            .add_metadata_u32("llama.attention.head_count_kv", 2)
            .add_metadata_u32("llama.feed_forward_length", 512)
            .add_metadata_u32("llama.vocab_size", 256)
            .add_metadata_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .add_metadata_f32("llama.rope.freq_base", 10000.0)
            .add_metadata_u32("llama.context_length", 2048)
            .add_tensor("token_embd.weight", &[256], GGML_TYPE_F32, &f32_bytes);
        let bytes = builder.build();
        let gguf = parse_from_bytes(&bytes).expect("parse");

        let config = parse_model_config(&gguf).unwrap();
        let source = GgufSource {
            gguf,
            config,
            weight_dtype: DType::Q4_0,
            cpu_backend: Arc::new(CpuBackend::new()),
        };

        let tensor = source
            .load_raw_cpu("token_embd.weight", false)
            .expect("F32 embed should load via F32 → F16 downcast path");

        assert_eq!(
            tensor.buffer().dtype(),
            DType::F16,
            "F32-stored embed must be downcast to F16 at load time"
        );
        assert_eq!(tensor.shape().numel(), 256);

        // Byte size must be 512 (256 × 2), NOT 1024 (256 × 4). This is the
        // RSS optimization the feature is named after.
        assert_eq!(tensor.buffer().size(), 256 * 2);

        // Spot-check values. 0.0, 0.5, 1.0, 1.5, ... are all exact in F16.
        let ptr = tensor.buffer().as_ptr() as *const u16;
        for i in 0..256 {
            let expected = (i as f32) * 0.5;
            let bits = unsafe { *ptr.add(i) };
            let got = f16::from_bits(bits).to_f32();
            assert_eq!(
                got, expected,
                "element {} mismatch: got {}, expected {}",
                i, got, expected
            );
        }
    }

    /// F32-stored non-embed tensors must continue to use the zero-copy
    /// `MmapBuffer` F32 path. The F16 downcast is strictly scoped to
    /// embed/lm_head because it is the only place where (a) the tensor is
    /// large enough to matter for RSS and (b) the downstream consumers
    /// (`gather`, final matmul) have validated F16 kernels.
    #[test]
    fn test_gguf_f32_non_embed_stays_f32() {
        let mut f32_bytes = Vec::<u8>::with_capacity(16 * 4);
        for i in 0..16 {
            f32_bytes.extend_from_slice(&(i as f32).to_le_bytes());
        }

        let mut builder = GgufTestBuilder::new();
        builder
            .add_metadata_str("general.architecture", "llama")
            .add_metadata_u32("llama.embedding_length", 16)
            .add_metadata_u32("llama.block_count", 1)
            .add_metadata_u32("llama.attention.head_count", 4)
            .add_metadata_u32("llama.attention.head_count_kv", 2)
            .add_metadata_u32("llama.feed_forward_length", 32)
            .add_metadata_u32("llama.vocab_size", 16)
            .add_metadata_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .add_metadata_f32("llama.rope.freq_base", 10000.0)
            .add_metadata_u32("llama.context_length", 2048)
            // Realistic F32 non-embed: a norm weight.
            .add_tensor("blk.0.attn_norm.weight", &[16], GGML_TYPE_F32, &f32_bytes);
        let bytes = builder.build();
        let gguf = parse_from_bytes(&bytes).expect("parse");

        let config = parse_model_config(&gguf).unwrap();
        let source = GgufSource {
            gguf,
            config,
            weight_dtype: DType::Q4_0,
            cpu_backend: Arc::new(CpuBackend::new()),
        };

        let tensor = source
            .load_raw_cpu("blk.0.attn_norm.weight", false)
            .expect("F32 norm should load via zero-copy MmapBuffer path");

        assert_eq!(
            tensor.buffer().dtype(),
            DType::F32,
            "non-embed F32 tensors must stay F32 (no unscoped downcast)"
        );
        assert_eq!(tensor.shape().numel(), 16);
        assert_eq!(tensor.buffer().size(), 16 * 4);
    }
}
