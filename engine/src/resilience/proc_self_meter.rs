//! Engine process 자신의 CPU 사용률 측정 (MSG-067, INV-091, INV-092).
//!
//! `/proc/self/stat`의 `utime` + `stime` jiffies 누적값을 샘플링 간 delta로
//! 나누어 `[0.0, 1.0]` 범위의 CPU 사용률을 산출한다. Heartbeat(MSG-060)의
//! `self_cpu_pct` 필드에 실어 Manager에 전달된다. Linux 외 플랫폼 또는 측정
//! 실패 시 0.0으로 fallback하며, Heartbeat 송출을 절대 차단하지 않는다.
//!
//! 분모: `CLK_TCK × num_cpus × wall_clock_delta_secs`
//! - `CLK_TCK`: `libc::sysconf(_SC_CLK_TCK)` (보통 100 Hz)
//! - `num_cpus`: `libc::sysconf(_SC_NPROCESSORS_ONLN)`
//!
//! 첫 샘플은 이전 값이 없으므로 0.0을 반환한다 (warm-up).

use std::fs;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
struct CpuSample {
    jiffies: u64,
    at: Instant,
}

/// Engine process self-CPU 사용률 측정기. Heartbeat 주기마다 `sample()`을
/// 호출하여 직전 샘플과의 delta 기반 사용률을 얻는다.
pub struct ProcSelfMeter {
    clk_tck: f64,
    num_cpus: f64,
    last: Option<CpuSample>,
}

impl ProcSelfMeter {
    /// 시스템 상수를 읽어 새 meter를 만든다. `CLK_TCK`/`NPROCESSORS_ONLN`
    /// 읽기 실패 시에도 생성에는 성공하며 (기본값 사용), 이후 `sample()`은
    /// 0.0을 반환한다.
    pub fn new() -> Self {
        let clk_tck = sysconf(libc::_SC_CLK_TCK).unwrap_or(100) as f64;
        let num_cpus = sysconf(libc::_SC_NPROCESSORS_ONLN).unwrap_or(1).max(1) as f64;
        Self {
            clk_tck,
            num_cpus,
            last: None,
        }
    }

    /// 현재 사용률을 반환한다. 범위 `[0.0, 1.0]`로 clamp 된다 (INV-091).
    /// 어떤 이유로든 측정 불가 시 0.0을 반환한다 (INV-092).
    pub fn sample(&mut self) -> f64 {
        let Some(jiffies) = read_proc_self_jiffies() else {
            return 0.0;
        };
        let now = Instant::now();
        let pct = match self.last {
            None => 0.0,
            Some(prev) => {
                let wall = now.duration_since(prev.at).as_secs_f64();
                if wall <= 0.0 {
                    0.0
                } else {
                    let delta = jiffies.saturating_sub(prev.jiffies) as f64;
                    let denom = self.clk_tck * self.num_cpus * wall;
                    if denom > 0.0 { delta / denom } else { 0.0 }
                }
            }
        };
        self.last = Some(CpuSample { jiffies, at: now });
        clamp_unit(pct)
    }
}

impl Default for ProcSelfMeter {
    fn default() -> Self {
        Self::new()
    }
}

fn clamp_unit(x: f64) -> f64 {
    if !x.is_finite() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

fn sysconf(name: libc::c_int) -> Option<libc::c_long> {
    let v = unsafe { libc::sysconf(name) };
    if v < 0 { None } else { Some(v) }
}

/// `/proc/self/stat`에서 `utime + stime` jiffies 합을 읽는다. 파싱 실패 시
/// `None`을 반환한다. comm 필드는 공백/괄호를 포함할 수 있으므로 `rfind(')')`
/// 이후부터 파싱해야 한다.
fn read_proc_self_jiffies() -> Option<u64> {
    let raw = fs::read_to_string("/proc/self/stat").ok()?;
    parse_utime_stime(&raw)
}

fn parse_utime_stime(stat: &str) -> Option<u64> {
    let rparen = stat.rfind(')')?;
    let tail = stat.get(rparen + 1..)?.trim_start();
    let fields: Vec<&str> = tail.split_ascii_whitespace().collect();
    // 괄호 이후 1번째 필드가 state(3번째 전체 필드). utime은 14번째, stime은 15번째.
    // 괄호 이후 인덱스로는 utime=11, stime=12 (0-based, state=0, ppid=1, ...).
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(utime.saturating_add(stime))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_stat(utime: u64, stime: u64) -> String {
        // 실제 /proc/self/stat 레이아웃 모사. comm에 공백/괄호 삽입으로 robust 검증.
        // 괄호 이후 필드 순서: state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt utime stime ...
        format!(
            "1234 (weird )name (2)) R 1 1 1 0 -1 0 0 0 0 0 {utime} {stime} 0 0 20 0 1 0 12345 0 0 ...\n"
        )
    }

    #[test]
    fn parse_utime_stime_handles_parens_in_comm() {
        let s = synth_stat(123, 45);
        assert_eq!(parse_utime_stime(&s), Some(168));
    }

    #[test]
    fn parse_utime_stime_returns_none_on_malformed_input() {
        assert_eq!(parse_utime_stime("no closing paren here"), None);
        assert_eq!(parse_utime_stime(") too few fields"), None);
    }

    #[test]
    fn clamp_unit_bounds() {
        assert_eq!(clamp_unit(-0.1), 0.0);
        assert_eq!(clamp_unit(1.5), 1.0);
        assert_eq!(clamp_unit(0.42), 0.42);
        assert_eq!(clamp_unit(f64::NAN), 0.0);
        assert_eq!(clamp_unit(f64::INFINITY), 0.0);
    }

    #[test]
    fn first_sample_returns_zero() {
        let mut m = ProcSelfMeter::new();
        assert_eq!(m.sample(), 0.0);
    }

    #[test]
    fn second_sample_is_in_unit_range() {
        let mut m = ProcSelfMeter::new();
        let _ = m.sample();
        // busy spin 짧게
        let start = std::time::Instant::now();
        while start.elapsed().as_millis() < 10 {
            std::hint::black_box(0);
        }
        let v = m.sample();
        assert!((0.0..=1.0).contains(&v), "expected 0..=1, got {v}");
    }
}
