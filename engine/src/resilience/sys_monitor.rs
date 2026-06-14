use anyhow::{Result, anyhow};
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    pub total: usize,
    pub available: usize,
    pub free: usize,
}

pub trait SystemMonitor: Send + Sync {
    fn mem_stats(&self) -> Result<MemoryStats>;
}

pub struct LinuxSystemMonitor;

impl Default for LinuxSystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl LinuxSystemMonitor {
    pub fn new() -> Self {
        Self
    }

    fn parse_meminfo(content: &str) -> Result<MemoryStats> {
        let mut total = None;
        let mut available = None;
        let mut free = None;

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let value = parts[1].parse::<usize>()?; // KB
            // meminfo uses KB, we want Bytes? Or just keep KB. Let's keep KB for internal, but maybe convert to bytes?
            // "MemTotal:       32768480 kB"
            // Let's standardise on Bytes.
            let value_bytes = value * 1024;

            match parts[0] {
                "MemTotal:" => total = Some(value_bytes),
                "MemAvailable:" => available = Some(value_bytes),
                "MemFree:" => free = Some(value_bytes),
                _ => {}
            }
        }

        Ok(MemoryStats {
            total: total.ok_or(anyhow!("MemTotal not found"))?,
            available: available.ok_or(anyhow!("MemAvailable not found"))?,
            free: free.ok_or(anyhow!("MemFree not found"))?,
        })
    }
}

impl SystemMonitor for LinuxSystemMonitor {
    fn mem_stats(&self) -> Result<MemoryStats> {
        let file = File::open("/proc/meminfo")?;
        let _reader = BufReader::new(file);

        // Read file into string for parsing (or stream parsing)
        // Since /proc/meminfo is small, reading to string is fine.
        let _content = String::new();
        // We only need the first few lines essentially, but reading all is safe.
        // Actually BufReader is good.
        // Let's reuse the static parsing function to make it testable
        // But we need to read it first.

        let content = std::fs::read_to_string("/proc/meminfo")?;
        Self::parse_meminfo(&content)
    }
}

/// No-op monitor that always reports abundant memory.
///
/// Used when system RAM metrics are unreliable — e.g., CUDA discrete GPU where
/// managed memory (cuMemAllocManaged) reserves system RAM for virtual address
/// space even though data resides in VRAM. Reading /proc/meminfo MemAvailable
/// would show artificially low values, triggering false memory pressure alerts.
///
/// With this monitor, KV cache eviction is driven purely by policy budgets
/// (e.g., H2O keep_ratio, sliding window size) rather than system RAM pressure.
pub struct NoOpMonitor;

impl SystemMonitor for NoOpMonitor {
    fn mem_stats(&self) -> Result<MemoryStats> {
        Ok(MemoryStats {
            total: usize::MAX,
            available: usize::MAX,
            free: usize::MAX,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linux_monitor_parsing() {
        let mock_meminfo = r#"
MemTotal:       32768480 kB
MemFree:         4567890 kB
MemAvailable:   12345678 kB
Buffers:          123456 kB
"#;
        let stats = LinuxSystemMonitor::parse_meminfo(mock_meminfo).unwrap();

        assert_eq!(stats.total, 32768480 * 1024);
        assert_eq!(stats.available, 12345678 * 1024);
        assert_eq!(stats.free, 4567890 * 1024);
    }

    #[test]
    fn test_parse_meminfo_missing_field_error() {
        // MemAvailable is missing → should error
        let incomplete = "MemTotal:  1024 kB\nMemFree:  512 kB\n";
        let result = LinuxSystemMonitor::parse_meminfo(incomplete);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("MemAvailable"),
            "expected MemAvailable error, got: {}",
            msg
        );
    }

    #[test]
    fn test_parse_meminfo_bad_format_error() {
        // Non-numeric value should cause parse error
        let bad = "MemTotal:  not_a_number kB\nMemFree:  512 kB\nMemAvailable:  256 kB\n";
        let result = LinuxSystemMonitor::parse_meminfo(bad);
        assert!(result.is_err());
    }
}
