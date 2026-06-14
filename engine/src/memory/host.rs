//! Host-managed memory resources (L2).
//!
//! - [`shared`]: heap `Vec<u8>` backed `SharedBuffer` + `SharedBufferView`.
//! - [`mmap`]: `MmapBuffer` (read-only mmap view) + `MmapKeepAlive` lifetime
//!   guard trait (INV-143).
pub mod mmap;
pub mod shared;
