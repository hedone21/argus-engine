//! `OpaqueBuffer` — `DType`-우회 opaque KV format 버퍼.
//!
//! 설계 SSOT: `docs/adr/0007-opaque-dtype-kv-format-unlock.md`.
//!
//! 새 KV format(`.so` block-quant family)은 새 `DType` enum variant 가 아니라
//! `OpaqueBuffer`(raw bytes + [`KVLayoutDesc`] sidecar) + `OpaqueKvFormat` impl 로 표현된다.
//! 엔진은 opaque 데이터를 **raw bytes 로만** 보고 descriptor floor(`dequant_via_descriptor`)로만
//! 읽는다 — typed accessor(`as_slice::<T>`)를 노출하지 않아 closed `DType` 로의 재해석을 차단한다.
//!
//! **D3 tag**: `Buffer::dtype()` 가 `DType` 반환을 강제하나 opaque 데이터는 closed `DType` 중
//! 무엇도 아니다. `dtype()` 는 `DType::U8`(raw bytes)을 반환하고 의미는 [`Self::descriptor`]
//! sidecar 가 운반한다. U8 arm 은 backend match 전반에서 이미 floor/Err 로 빠지므로 typed
//! 재해석이 자연 차단된다(`buffer.rs` enum 무변 — 외과적).
//!
//! 저장은 inner `Arc<dyn Buffer>`(CPU=SharedBuffer, 향후 GPU=UnifiedBuffer/OpenCLBuffer)에
//! 위임한다 — opaque 는 그 위에 U8 재태깅 + descriptor 부착만 한다.

use std::any::Any;
use std::sync::Arc;

use anyhow::Result;
use argus_extension_api::KVLayoutDesc;
#[cfg(feature = "opencl")]
use ocl::core::Mem;

use crate::buffer::{Buffer, DType};

/// raw 바이트 저장 + `KVLayoutDesc` sidecar 를 묶는 opaque KV format 버퍼.
pub struct OpaqueBuffer {
    /// raw 바이트 저장 (U8-typed inner). ptr/size/sync/map 은 전부 여기 위임.
    inner: Arc<dyn Buffer>,
    /// 이 버퍼의 진짜 format 의미 — `Buffer::dtype()`(U8)가 못 담는 정보.
    desc: KVLayoutDesc,
}

impl OpaqueBuffer {
    /// inner 저장 버퍼를 descriptor 와 함께 opaque 로 wrapping. inner 는 raw 바이트 컨테이너
    /// (보통 `DType::U8` 로 alloc 된 SharedBuffer) — alloc 은 호출자(KVCache) 책임.
    pub fn new(inner: Arc<dyn Buffer>, desc: KVLayoutDesc) -> Self {
        Self { inner, desc }
    }

    /// 이 버퍼의 format descriptor (sidecar). byte-회계·dequant floor 가 dtype 대신 이걸 읽는다.
    pub fn descriptor(&self) -> KVLayoutDesc {
        self.desc
    }

    /// inner 저장의 raw 바이트 (packed block 연속). 엔진은 이걸 descriptor floor 로만 해석한다.
    pub fn raw_bytes(&self) -> &[u8] {
        // SAFETY: inner.as_ptr() 은 inner.size() 바이트에 대해 유효하고, inner(Arc) 가 self
        // 수명 동안 살아 있으므로 반환 슬라이스의 수명은 &self 에 묶인다.
        unsafe { std::slice::from_raw_parts(self.inner.as_ptr(), self.inner.size()) }
    }

    /// inner 저장 버퍼 핸들 (Tensor 생성·GPU 핸들 접근용).
    pub fn inner(&self) -> &Arc<dyn Buffer> {
        &self.inner
    }
}

impl Buffer for OpaqueBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// opaque tag. 진짜 의미는 [`Self::descriptor`] sidecar.
    fn dtype(&self) -> DType {
        DType::U8
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn as_ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.inner.as_mut_ptr()
    }

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&Mem> {
        self.inner.cl_mem()
    }

    #[cfg(not(feature = "opencl"))]
    fn cl_mem(&self) -> Option<()> {
        self.inner.cl_mem()
    }

    fn sync_device(&self) -> Result<()> {
        self.inner.sync_device()
    }

    fn map_for_cpu(&self) -> Result<()> {
        self.inner.map_for_cpu()
    }

    fn unmap_for_gpu(&self) -> Result<()> {
        self.inner.unmap_for_gpu()
    }

    fn is_mapped(&self) -> bool {
        self.inner.is_mapped()
    }

    fn is_host_managed(&self) -> bool {
        self.inner.is_host_managed()
    }

    fn is_gpu_buffer(&self) -> bool {
        self.inner.is_gpu_buffer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::host::shared::SharedBuffer;
    use argus_extension_api::{Packing, ScaleLayout};

    fn q4_0_desc() -> KVLayoutDesc {
        KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        }
    }

    #[test]
    fn opaque_buffer_tags_u8_and_carries_descriptor() {
        // q4_0 layout 1 블록 = 18 바이트.
        let nbytes = q4_0_desc().bytes_for_elems(32).unwrap();
        assert_eq!(nbytes, 18);
        let inner: Arc<dyn Buffer> = Arc::new(SharedBuffer::new(nbytes, DType::U8));
        let opaque = OpaqueBuffer::new(inner, q4_0_desc());

        // D3: dtype 은 U8 tag, 의미는 descriptor sidecar.
        assert_eq!(opaque.dtype(), DType::U8);
        assert_eq!(opaque.descriptor(), q4_0_desc());
        assert_eq!(opaque.size(), 18);
        assert_eq!(opaque.raw_bytes().len(), 18);
    }

    #[test]
    fn opaque_buffer_downcasts_from_dyn() {
        let inner: Arc<dyn Buffer> = Arc::new(SharedBuffer::new(18, DType::U8));
        let dyn_buf: Arc<dyn Buffer> = Arc::new(OpaqueBuffer::new(inner, q4_0_desc()));
        // byte-회계·floor 가 opaque 를 식별하는 경로: as_any().downcast_ref.
        let recovered = dyn_buf
            .as_any()
            .downcast_ref::<OpaqueBuffer>()
            .expect("dyn Buffer downcasts back to OpaqueBuffer");
        assert_eq!(recovered.descriptor(), q4_0_desc());
    }
}
