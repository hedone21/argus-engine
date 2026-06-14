use crate::buffer::{Buffer, DType};
use crate::memory::Memory;
use crate::memory::host::shared::SharedBuffer;
use anyhow::Result;
use std::sync::Arc;

pub struct Galloc;

impl Galloc {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Galloc {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory for Galloc {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        let buf = SharedBuffer::new(size, dtype);
        Ok(Arc::new(buf))
    }

    fn used_memory(&self) -> usize {
        0 // Tracking not implemented
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_galloc_allocation() {
        let allocator = Galloc::new();

        let buffer_result = allocator.alloc(1024, DType::F32);
        assert!(buffer_result.is_ok());

        let buffer = buffer_result.unwrap();
        assert_eq!(buffer.size(), 1024);
        assert_eq!(buffer.dtype(), DType::F32);

        let ptr = buffer.as_ptr();
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_galloc_zero_size_allocation() {
        let allocator = Galloc::new();
        let buffer = allocator.alloc(0, DType::U8).unwrap();
        assert_eq!(buffer.size(), 0);
        assert_eq!(buffer.dtype(), DType::U8);
    }

    #[test]
    fn test_galloc_used_memory() {
        let allocator = Galloc::new();
        // Since tracking is not implemented, it currently returns 0
        assert_eq!(allocator.used_memory(), 0);
        let _buf = allocator.alloc(1024, DType::F32).unwrap();
        assert_eq!(allocator.used_memory(), 0);
    }
}
