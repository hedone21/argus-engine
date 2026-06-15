#!/usr/bin/env python3
"""Unified device runner — build, deploy, and execute binaries on any registered device.

Usage:
    python scripts/run_device.py -d android argus-cli --prompt "Hello" -n 128
    python scripts/run_device.py -d host test_backend
    python scripts/run_device.py -d android --skip-build argus-cli -b opencl
    python scripts/run_device.py --list-devices

    # Build + deploy multiple binaries without executing (deploy-only):
    python scripts/run_device.py -d android --skip-exec argus-cli --extra-bin test_backend
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from device_registry.builder import build_binary, get_local_binary_path
from device_registry.config import load_all_devices, load_device_config
from device_registry.connection import create_connection
from device_registry.deployer import deploy_binary, deploy_eval_files, verify_model

# Binaries that consume `--model-path` and benefit from model verification.
_INFERENCE_BINARIES = ("argus-cli", "argus-eval")


def _list_devices() -> int:
    devices = load_all_devices()
    if not devices:
        print("No devices in devices.toml")
        return 1
    print(f"{'ID':<15} {'Name':<25} {'Type':<8} {'Target'}")
    print("-" * 70)
    for dev_id, cfg in devices.items():
        target = cfg.build.target or "(native)"
        print(f"{dev_id:<15} {cfg.name:<25} {cfg.connection.type:<8} {target}")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified device runner",
        usage="%(prog)s [-d DEVICE] [--skip-build] [--dry-run] BINARY [ARGS...]",
    )
    parser.add_argument("-d", "--device", default="host", help="Device ID (default: host)")
    parser.add_argument("--skip-build", action="store_true", help="Skip build step")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip deploy step")
    parser.add_argument("--skip-exec", action="store_true", help="Build + deploy only, no execute")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--list-devices", action="store_true", help="List registered devices")
    parser.add_argument("--deploy-eval", action="store_true", help="Deploy experiments/prompts/ files too")
    parser.add_argument("--extra-bin", action="append", default=[], help="Additional binaries to build")

    # Parse known args, rest goes to the binary
    args, remaining = parser.parse_known_args()

    if args.list_devices:
        return _list_devices()

    if not remaining:
        parser.print_help()
        print("\nError: BINARY name required (e.g., argus-cli, test_backend)")
        return 1

    binary_name = remaining[0]
    binary_args = remaining[1:]

    # Load device config
    try:
        device = load_device_config(args.device)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Device: {device.name} ({device.id}, {device.connection.type})")

    # 1. Build
    if not args.skip_build:
        print("\n[1/3] Building...")
        ok = build_binary(
            device, binary_name, _PROJECT_ROOT,
            extra_bins=args.extra_bin or None,
            dry_run=args.dry_run,
        )
        if not ok:
            print("Build failed!")
            return 1
    else:
        print("\n[1/3] Build skipped")

    # 2. Deploy
    if not args.skip_deploy:
        print("\n[2/3] Deploying...")
        local_bin = get_local_binary_path(device, binary_name, _PROJECT_ROOT)
        conn = create_connection(device.connection)

        ok = deploy_binary(conn, device, local_bin, binary_name, dry_run=args.dry_run)
        if not ok:
            print("Deploy failed!")
            return 1

        if args.deploy_eval:
            deploy_eval_files(conn, device, _PROJECT_ROOT / "experiments" / "prompts", dry_run=args.dry_run)

        # Verify model if it's an inference binary
        if binary_name in _INFERENCE_BINARIES and not args.dry_run:
            verify_model(conn, device)
    else:
        print("\n[2/3] Deploy skipped")
        conn = create_connection(device.connection)

    # 2.5 skip-exec shortcut
    if args.skip_exec:
        print("\n[3/3] Skipped (--skip-exec)")
        return 0

    # 3. Execute
    print("\n[3/3] Executing...")

    # Build the command
    if device.is_local:
        local_bin = get_local_binary_path(device, binary_name, _PROJECT_ROOT)
        cmd = f"{local_bin} {' '.join(binary_args)}"

        # Add model-path if not explicitly provided and we have a model_dir
        if device.paths.model_dir and "--model-path" not in cmd:
            cmd = f"{local_bin} --model-path {device.paths.model_dir} {' '.join(binary_args)}"
    else:
        remote_bin = device.binary_remote_path(binary_name)
        args_str = " ".join(binary_args)

        # Add model-path for inference binaries if not explicitly provided
        if (
            device.paths.model_dir
            and "--model-path" not in args_str
            and binary_name in _INFERENCE_BINARIES
        ):
            args_str = f"--model-path {device.paths.model_dir} {args_str}"

        cmd = f"{remote_bin} {args_str}"

    env_vars = {}
    if device.paths.lib_dir:
        env_vars["LD_LIBRARY_PATH"] = device.paths.lib_dir

    print(f"  Command: {cmd}")
    if env_vars:
        print(f"  Env: {env_vars}")

    if args.dry_run:
        print("  (dry-run: skipping execution)")
        return 0

    result = conn.execute(cmd, env_vars=env_vars)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
