//! Signal injector for resilience stress testing.
//!
//! Listens on a Unix socket or TCP address, waits for the generate binary to
//! connect, then sends ManagerMessage directives according to a JSON schedule
//! file.
//!
//! Schedule format:
//! ```json
//! [
//!   {"delay_sec": 10, "directive": {"seq_id": 1, "commands": [{"type": "set_memory_level", "level": "critical", "target_ratio": 0.5}]}},
//!   {"delay_sec": 20, "directive": {"seq_id": 2, "commands": [{"type": "suspend"}]}}
//! ]
//! ```
//!
//! Transport selection:
//! - `--socket tcp:127.0.0.1:9100` — TCP loopback (Android SELinux safe)
//! - `--socket /tmp/llm.sock`      — Unix domain socket (default)

use argus_shared::{EngineDirective, ManagerMessage};
use clap::Parser;
use serde::Deserialize;
use std::io::Write;
use std::net::TcpListener;
#[cfg(unix)]
use std::os::unix::net::UnixListener;

#[derive(Parser, Debug)]
#[command(about = "Inject resilience directives via Unix socket or TCP for stress testing")]
struct Args {
    /// Transport address. Prefix with "tcp:" for TCP (e.g. "tcp:127.0.0.1:9100"),
    /// or provide a bare path for a Unix domain socket (e.g. "/tmp/llm.sock").
    #[arg(short, long)]
    socket: String,

    /// Path to the JSON schedule file
    #[arg(short = 'f', long)]
    schedule_file: std::path::PathBuf,

    /// Timeout in seconds to wait for a client connection
    #[arg(long, default_value_t = 30)]
    connect_timeout: u64,
}

#[derive(Debug, Deserialize)]
struct ScheduleEntry {
    delay_sec: f64,
    directive: EngineDirective,
}

fn write_message<W: Write>(writer: &mut W, msg: &ManagerMessage) -> std::io::Result<()> {
    let json = serde_json::to_vec(msg).expect("ManagerMessage serialization should not fail");
    let len = (json.len() as u32).to_be_bytes();
    writer.write_all(&len)?;
    writer.write_all(&json)?;
    writer.flush()
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Read schedule
    let schedule_json = std::fs::read_to_string(&args.schedule_file)?;
    let schedule: Vec<ScheduleEntry> = serde_json::from_str(&schedule_json)?;
    eprintln!(
        "[Injector] Loaded {} directives from {}",
        schedule.len(),
        args.schedule_file.display()
    );

    // Dispatch to TCP or Unix listener
    if args.socket.starts_with("tcp:") {
        let addr = &args.socket[4..];
        run_tcp(addr, &schedule, args.connect_timeout)
    } else {
        #[cfg(unix)]
        {
            run_unix(&args.socket, &schedule, args.connect_timeout)
        }
        #[cfg(not(unix))]
        {
            anyhow::bail!(
                "Unix domain sockets are not supported on this platform. Use tcp: prefix."
            );
        }
    }
}

fn run_tcp(addr: &str, schedule: &[ScheduleEntry], connect_timeout: u64) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr)?;
    listener.set_nonblocking(true)?;
    eprintln!(
        "[Injector] Listening on tcp:{} (timeout {}s)",
        addr, connect_timeout
    );

    let start = std::time::Instant::now();
    let mut stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if start.elapsed().as_secs() >= connect_timeout {
                    anyhow::bail!("Timed out waiting for client connection");
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => return Err(e.into()),
        }
    };
    stream.set_nonblocking(false)?;
    eprintln!("[Injector] Client connected");

    send_schedule(&mut stream, schedule);
    Ok(())
}

#[cfg(unix)]
fn run_unix(path: &str, schedule: &[ScheduleEntry], connect_timeout: u64) -> anyhow::Result<()> {
    let _ = std::fs::remove_file(path);
    let listener = UnixListener::bind(path)?;
    listener.set_nonblocking(false)?;
    eprintln!(
        "[Injector] Listening on {} (timeout {}s)",
        path, connect_timeout
    );

    listener.set_nonblocking(true).ok();
    let start = std::time::Instant::now();
    let mut stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if start.elapsed().as_secs() >= connect_timeout {
                    anyhow::bail!("Timed out waiting for client connection");
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => return Err(e.into()),
        }
    };
    stream.set_nonblocking(false)?;
    eprintln!("[Injector] Client connected");

    send_schedule(&mut stream, schedule);
    let _ = std::fs::remove_file(path);
    Ok(())
}

fn send_schedule<W: Write>(writer: &mut W, schedule: &[ScheduleEntry]) {
    for (i, entry) in schedule.iter().enumerate() {
        if entry.delay_sec > 0.0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(entry.delay_sec));
        }

        let msg = ManagerMessage::Directive(entry.directive.clone());
        match write_message(writer, &msg) {
            Ok(()) => {
                eprintln!(
                    "[Injector] [{}/{}] Sent: seq_id={}, {} commands",
                    i + 1,
                    schedule.len(),
                    entry.directive.seq_id,
                    entry.directive.commands.len()
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {
                eprintln!("[Injector] Client disconnected (broken pipe)");
                return;
            }
            Err(e) => {
                eprintln!("[Injector] Send error: {}", e);
                return;
            }
        }
    }
    eprintln!("[Injector] Schedule complete, exiting");
}
