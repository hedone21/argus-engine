//! Quantize handler — reduces KV cache precision to save memory.
//!
//! Works with `KiviCache` to dynamically transition quantization bit-width
//! based on pressure level. Maps pressure levels to target bits:
//! - Normal: no action (keep current bits)
//! - Warning: 8-bit
//! - Critical: 4-bit
//! - Emergency: 2-bit

use super::PressureLevel;

/// Map pressure level to target KIVI quantization bits. (ENG-ALG-092)
/// Returns `None` if no transition is needed (Normal pressure).
/// production 호출지 없음 — KIVI dynamic transition 미배선. spec MUST 보존 진입점.
pub fn target_bits_for_pressure(level: PressureLevel) -> Option<u8> {
    match level {
        PressureLevel::Normal => None,
        PressureLevel::Warning => Some(8),
        PressureLevel::Critical => Some(4),
        PressureLevel::Emergency => Some(2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::PressureLevel;

    #[test]
    fn test_target_bits_normal() {
        assert_eq!(target_bits_for_pressure(PressureLevel::Normal), None);
    }

    #[test]
    fn test_target_bits_warning() {
        assert_eq!(target_bits_for_pressure(PressureLevel::Warning), Some(8));
    }

    #[test]
    fn test_target_bits_critical() {
        assert_eq!(target_bits_for_pressure(PressureLevel::Critical), Some(4));
    }

    #[test]
    fn test_target_bits_emergency() {
        assert_eq!(target_bits_for_pressure(PressureLevel::Emergency), Some(2));
    }
}
