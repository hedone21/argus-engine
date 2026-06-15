# Demo recording

`plugin.gif` (embedded in the README) shows the **StreamingLLM KV-cache eviction stage**
running on a phone's Adreno GPU. It is a side-by-side of two on-device
`argus-chat --interactive` sessions over OpenCL, both capped at `--max-seq-len 512`:

- **left — `eviction none`**: a multi-turn chat fills the KV cache and **overflows at the
  sequence-length limit**, so generation stops.
- **right — `eviction streaming --sink 4 --recent-window 256`**: the StreamingLLM stage
  **prunes the cache each turn** (`[Chat/Evict] removed=… new_pos=…`) and the chat keeps
  going.

## How it was recorded

On-device (Galaxy S25 / Adreno 830) — the engine's primary OpenCL target. The flow:

1. Cross-compile + deploy: `python scripts/run_device.py -d android --skip-exec argus-chat`
   (needs `hosts.toml` + `devices.toml`; see the `.example` templates), plus a GGUF model
   pushed to `/data/local/tmp/models/`.
2. Run on-device via `adb shell` with `LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64`,
   driving the REPL turns through the device's interactive shell.
3. Record each side with [asciinema](https://asciinema.org) and render to GIF with
   [agg](https://github.com/asciinema/agg) (pure-Rust, no headless browser). The long
   model-load / prefill pauses are compressed with `agg --idle-time-limit`.
4. Compose the two halves side-by-side (with the caption banners) using `ffmpeg`
   `hstack` + `palettegen`/`paletteuse`.

Recorded without `--profile`, per the repo's performance-measurement rule.
