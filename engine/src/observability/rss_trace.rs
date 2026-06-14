//! Lightweight RSS (Resident Set Size) tracing utility.
//!
//! Activated only when `LLMRS_RSS_TRACE=1` is set. Reads `/proc/self/status`
//! and prints a single structured line to stderr.
//!
//! Output format:
//!   `[RSS] tag=<tag> VmRSS=<kb> RssFile=<kb> RssAnon=<kb> RssShmem=<kb> VmData=<kb> VmSize=<kb>`
//!
//! `dump_smaps(tag)`: controlled by `LLMRS_DUMP_SMAPS_T1`. Copies
//! `/proc/self/smaps` to a file under `LLMRS_DUMP_DIR` (default:
//! `/data/local/tmp/argus_engine/` on Android, `/tmp/` on Linux).
//!
//! On non-Linux/non-Android targets these are no-ops.

use std::sync::OnceLock;

/// Returns `true` if `LLMRS_RSS_TRACE` is set in the environment.
/// The result is cached after the first call.
fn rss_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLMRS_RSS_TRACE").is_ok())
}

fn io_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLMRS_IO_TRACE").is_ok())
}

/// Print a one-line RSS snapshot to stderr, tagged with `tag`.
///
/// When `LLMRS_RSS_TRACE` is not set, this function is a zero-cost no-op:
/// the `rss_trace_enabled()` check is cache-hit after the first call and the
/// function body is entirely skipped.
///
/// On non-Linux targets the function is always a no-op regardless of the
/// environment variable.
pub fn rss_trace(tag: &str) {
    if !rss_trace_enabled() {
        return;
    }
    rss_trace_impl(tag);
}

/// Linux / Android implementation: parse `/proc/self/status` and emit one line.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn rss_trace_impl(tag: &str) {
    match read_proc_status() {
        Ok(fields) => {
            eprintln!(
                "[RSS] tag={} VmRSS={} RssFile={} RssAnon={} RssShmem={} VmData={} VmSize={}",
                tag,
                fields.vm_rss,
                fields.rss_file,
                fields.rss_anon,
                fields.rss_shmem,
                fields.vm_data,
                fields.vm_size,
            );
        }
        Err(e) => {
            eprintln!("[RSS] tag={} error={}", tag, e);
        }
    }
}

/// Non-Linux/non-Android stub: always no-op.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn rss_trace_impl(_tag: &str) {}

/// Print a one-line `/proc/self/io` snapshot to stderr, tagged with `tag`.
/// Controlled by `LLMRS_IO_TRACE`. Used to verify whether prefault/swap
/// stages perform real disk read or only metadata operations.
pub fn io_trace(tag: &str) {
    if !io_trace_enabled() {
        return;
    }
    io_trace_impl(tag);
}

/// Return `read_bytes` from `/proc/self/io` (0 on error / non-Linux).
/// Used by per-phase delta tracing (e.g., swap_executor tick).
pub fn read_bytes_now() -> u64 {
    read_bytes_impl()
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn io_trace_impl(tag: &str) {
    match read_proc_io() {
        Ok(io) => eprintln!(
            "[IO] tag={} rchar={} wchar={} read_bytes={} write_bytes={}",
            tag, io.rchar, io.wchar, io.read_bytes, io.write_bytes,
        ),
        Err(e) => eprintln!("[IO] tag={} error={}", tag, e),
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn io_trace_impl(_tag: &str) {}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_bytes_impl() -> u64 {
    read_proc_io().map(|io| io.read_bytes).unwrap_or(0)
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn read_bytes_impl() -> u64 {
    0
}

#[cfg(any(target_os = "linux", target_os = "android"))]
struct ProcIoFields {
    rchar: u64,
    wchar: u64,
    read_bytes: u64,
    write_bytes: u64,
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_proc_io() -> Result<ProcIoFields, String> {
    let content = std::fs::read_to_string("/proc/self/io").map_err(|e| format!("read: {e}"))?;
    let mut rchar = 0u64;
    let mut wchar = 0u64;
    let mut read_bytes = 0u64;
    let mut write_bytes = 0u64;
    for line in content.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let val: u64 = rest.trim().parse().unwrap_or(0);
        match key.trim() {
            "rchar" => rchar = val,
            "wchar" => wchar = val,
            "read_bytes" => read_bytes = val,
            "write_bytes" => write_bytes = val,
            _ => {}
        }
    }
    Ok(ProcIoFields {
        rchar,
        wchar,
        read_bytes,
        write_bytes,
    })
}

// ---------------------------------------------------------------------------
// /proc/self/status parser (Linux and Android)
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "linux", target_os = "android"))]
struct ProcStatusFields {
    vm_rss: u64,
    rss_file: u64,
    rss_anon: u64,
    rss_shmem: u64,
    vm_data: u64,
    vm_size: u64,
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_proc_status() -> Result<ProcStatusFields, String> {
    let content = std::fs::read_to_string("/proc/self/status").map_err(|e| format!("read: {e}"))?;

    let mut vm_rss = 0u64;
    let mut rss_file = 0u64;
    let mut rss_anon = 0u64;
    let mut rss_shmem = 0u64;
    let mut vm_data = 0u64;
    let mut vm_size = 0u64;

    for line in content.lines() {
        // Lines look like:  "VmRSS:\t  12345 kB"
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let val: u64 = rest
            .split_whitespace()
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        match key.trim() {
            "VmRSS" => vm_rss = val,
            "RssFile" => rss_file = val,
            "RssAnon" => rss_anon = val,
            "RssShmem" => rss_shmem = val,
            "VmData" => vm_data = val,
            "VmSize" => vm_size = val,
            _ => {}
        }
    }

    Ok(ProcStatusFields {
        vm_rss,
        rss_file,
        rss_anon,
        rss_shmem,
        vm_data,
        vm_size,
    })
}

// ---------------------------------------------------------------------------
// smaps dump (Linux and Android)
// ---------------------------------------------------------------------------

/// Copy `/proc/self/smaps` to a diagnostics file.
///
/// Controlled by `LLMRS_DUMP_SMAPS_T1` environment variable.
/// Output directory: `LLMRS_DUMP_DIR` (default: `/data/local/tmp/argus_engine/` on
/// Android, `/tmp/` on Linux).
///
/// File name: `smaps_<tag>_<pid>_<timestamp_ms>.txt`
///
/// The Tester pulls this file to analyse kgsl/ion/dmabuf VMA distribution.
/// On non-Linux/non-Android targets this is always a no-op.
pub fn dump_smaps(tag: &str) {
    dump_smaps_impl(tag);
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn dump_smaps_impl(tag: &str) {
    let smaps_src = "/proc/self/smaps";

    // Determine output directory: LLMRS_DUMP_DIR > OS default
    let dump_dir = std::env::var("LLMRS_DUMP_DIR").unwrap_or_else(|_| {
        if cfg!(target_os = "android") {
            "/data/local/tmp/argus_engine".to_string()
        } else {
            "/tmp".to_string()
        }
    });

    // Ensure directory exists (best-effort; failure is non-fatal)
    let _ = std::fs::create_dir_all(&dump_dir);

    let pid = std::process::id();
    let ts_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let filename = format!("smaps_{tag}_{pid}_{ts_ms}.txt");
    let dest_path = format!("{dump_dir}/{filename}");

    match std::fs::copy(smaps_src, &dest_path) {
        Ok(bytes) => eprintln!("[smaps] dumped {bytes} bytes → {dest_path}"),
        Err(e) => eprintln!("[smaps] dump failed tag={tag}: {e}"),
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn dump_smaps_impl(_tag: &str) {}
