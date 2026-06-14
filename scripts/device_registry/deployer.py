"""Deploy orchestration — push binaries and files to devices."""

from __future__ import annotations

from pathlib import Path

from .config import DeviceConfig
from .connection import Connection


def deploy_binary(
    conn: Connection,
    device_config: DeviceConfig,
    local_binary_path: str | Path,
    binary_name: str,
    dry_run: bool = False,
) -> bool:
    """Push a binary to the device and make it executable.

    For local devices, this is a no-op (binary runs in-place).
    Returns True on success.
    """
    if device_config.is_local:
        return True

    local_binary_path = Path(local_binary_path)
    if not dry_run and not local_binary_path.exists():
        print(f"  Error: binary not found: {local_binary_path}")
        return False

    remote_path = device_config.binary_remote_path(binary_name)
    print(f"  Deploy: {local_binary_path} -> {remote_path}")

    if dry_run:
        print("  (dry-run: skipping deploy)")
        return True

    conn.mkdir(device_config.paths.work_dir)
    conn.push(local_binary_path, remote_path)
    conn.chmod(remote_path, "+x")
    return True


def deploy_eval_files(
    conn: Connection,
    device_config: DeviceConfig,
    local_eval_dir: str | Path,
    dry_run: bool = False,
) -> bool:
    """Push evaluation files to the device eval directory.

    Returns True on success.
    """
    if not device_config.paths.eval_dir:
        print("  Warning: no eval_dir configured, skipping eval deploy")
        return True

    local_eval_dir = Path(local_eval_dir)
    if not local_eval_dir.exists():
        print(f"  Warning: local eval dir not found: {local_eval_dir}")
        return False

    if device_config.is_local:
        return True

    print(f"  Deploy eval: {local_eval_dir} -> {device_config.paths.eval_dir}")

    if dry_run:
        print("  (dry-run: skipping eval deploy)")
        return True

    conn.mkdir(device_config.paths.eval_dir)
    for f in sorted(local_eval_dir.iterdir()):
        if f.is_file():
            remote = f"{device_config.paths.eval_dir}/{f.name}"
            conn.push(f, remote)

    return True


def verify_model(conn: Connection, device_config: DeviceConfig) -> bool:
    """Check that the model exists on the device.

    Accepts either a single-file asset (`.gguf` / `.auf` — the path itself
    exists) or a Safetensors directory (contains `config.json`).
    """
    model_dir = device_config.paths.model_dir
    if not model_dir:
        print("  Warning: no model_dir configured")
        return False

    exists = conn.file_exists(model_dir) or conn.file_exists(f"{model_dir}/config.json")
    if not exists:
        print(f"  Warning: model not found at {model_dir}")
    return exists
