#!/usr/bin/env python3
"""Run `argus_eval --eval-ll` in fixed-size chunks, restarting the process
between chunks so the GPU/OpenCL driver releases its state.

Some host OpenCL drivers (e.g. NVIDIA) accumulate deferred allocations across
a long evaluation and eventually fail with CL_OUT_OF_RESOURCES. Re-launching
the binary every N questions bounds that accumulation below the failure
threshold while keeping the GPU prefill path active.

Usage:
    python3 scripts/eval_ll_batched.py \\
        --binary ./target/release/argus_eval \\
        --model-path models/gemma3-4b \\
        --eval-batch /tmp/eval_batch_10q.json \\
        --output /tmp/eval_out.json \\
        --chunk-size 8 \\
        -- --backend opencl --kv-type f32 --max-seq-len 4096 \\
           --qcf-mode both --greedy

Positional args after `--` are forwarded to the argus_eval binary verbatim.
Env vars (OCL_PLATFORM, RUST_LOG, ...) are inherited from the caller.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def split_batch(batch_path: Path, chunk_size: int) -> list[list[dict]]:
    with batch_path.open() as f:
        entries = json.load(f)
    return [entries[i : i + chunk_size] for i in range(0, len(entries), chunk_size)]


def _invoke_generate(
    binary: Path,
    model_path: str,
    chunk_file: Path,
    forwarded_args: list[str],
) -> tuple[subprocess.CompletedProcess, float]:
    cmd = [
        str(binary),
        "--model-path", model_path,
        "--eval-ll",
        "--eval-batch", str(chunk_file),
        *forwarded_args,
    ]
    start = time.monotonic()
    proc = subprocess.run(cmd, env=os.environ.copy(), capture_output=True, text=True)
    return proc, time.monotonic() - start


def run_chunk(
    binary: Path,
    model_path: str,
    chunk_entries: list[dict],
    forwarded_args: list[str],
    work_dir: Path,
    chunk_idx: int,
    total_chunks: int,
    max_retries: int,
) -> dict:
    chunk_file = work_dir / f"chunk_{chunk_idx:03d}.json"
    chunk_file.write_text(json.dumps(chunk_entries))

    last_stderr = ""
    for attempt in range(1, max_retries + 2):  # 1 + max_retries total
        attempt_tag = f"[batch {chunk_idx + 1}/{total_chunks}"
        if attempt > 1:
            attempt_tag += f" retry {attempt - 1}/{max_retries}"
        attempt_tag += "]"
        print(f"{attempt_tag} {len(chunk_entries)} questions running…", flush=True)

        proc, elapsed = _invoke_generate(binary, model_path, chunk_file, forwarded_args)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()
            last_stderr = proc.stderr

        if proc.returncode != 0:
            print(
                f"{attempt_tag} FAILED (exit {proc.returncode}, {elapsed:.1f}s); "
                f"{'retrying' if attempt <= max_retries else 'giving up'}",
                flush=True,
            )
            continue

        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raw = work_dir / f"chunk_{chunk_idx:03d}_stdout_attempt{attempt}.txt"
            raw.write_text(proc.stdout)
            print(
                f"{attempt_tag} stdout not JSON ({e}); saved to {raw}; "
                f"{'retrying' if attempt <= max_retries else 'giving up'}",
                flush=True,
            )
            continue

        print(f"{attempt_tag} done ({elapsed:.1f}s)", flush=True)
        return parsed

    sys.exit(
        f"[batch {chunk_idx + 1}/{total_chunks}] failed after {max_retries + 1} attempts. "
        f"Last stderr tail:\n{last_stderr[-500:]}"
    )


def merge_outputs(chunk_outputs: list[dict]) -> dict:
    """Merge per-chunk EvalOutput JSONs into one. Concatenates `results`,
    sums wall_time_s, keeps the first chunk's config/aggregate fields."""
    if not chunk_outputs:
        return {}
    merged = dict(chunk_outputs[0])
    merged["results"] = []
    total_wall = 0.0
    for out in chunk_outputs:
        merged["results"].extend(out.get("results", []))
        total_wall += float(out.get("wall_time_s", 0.0))
    merged["wall_time_s"] = total_wall
    merged["n_questions"] = len(merged["results"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=Path("./target/release/argus_eval"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help=(
            "Per-chunk retries on failure. NVIDIA OpenCL driver state is "
            "non-deterministic — same input may fail once then pass. Defaults "
            "to 3 attempts before giving up."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Intermediate chunk files directory (default: tempdir, deleted on success).",
    )
    # Everything after `--` goes through to the binary.
    parser.add_argument("forwarded", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.binary.exists():
        sys.exit(f"binary not found: {args.binary}")
    if not args.eval_batch.exists():
        sys.exit(f"eval-batch not found: {args.eval_batch}")

    forwarded = args.forwarded
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    chunks = split_batch(args.eval_batch, args.chunk_size)
    if not chunks:
        sys.exit("eval-batch is empty")

    use_tmp = args.work_dir is None
    work_dir = Path(tempfile.mkdtemp(prefix="eval_ll_batched_")) if use_tmp else args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        outputs = []
        overall_start = time.monotonic()
        for i, chunk in enumerate(chunks):
            outputs.append(
                run_chunk(
                    args.binary,
                    args.model_path,
                    chunk,
                    forwarded,
                    work_dir,
                    i,
                    len(chunks),
                    args.max_retries,
                )
            )
        merged = merge_outputs(outputs)
        args.output.write_text(json.dumps(merged, indent=2))
        total = time.monotonic() - overall_start
        print(
            f"\nMerged {sum(len(o.get('results', [])) for o in outputs)} results "
            f"→ {args.output} (total wall {total:.1f}s, {len(chunks)} chunks)"
        )
    finally:
        if use_tmp:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
