"""Build orchestration for device targets."""

from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path

from .config import DeviceConfig, HostConfig, load_hosts_config, detect_current_host

# Module-level flag to emit DeprecationWarning only once per process
_warned_env_file: bool = False


class ToolchainNotFoundError(Exception):
    """Raised when a required toolchain binary cannot be found."""


def _compose_toolchain_env(
    host: HostConfig,
    toolchain_name: str,
    cargo_target: str | None,
) -> dict[str, str]:
    """Compose CC/CXX/AR/LINKER env vars from a toolchain entry.

    Returns a new dict (never mutates os.environ).
    """
    if toolchain_name not in host.toolchains:
        available = list(host.toolchains.keys())
        raise KeyError(
            f"Toolchain {toolchain_name!r} not found in host {host.id!r}. "
            f"Available: {available}"
        )

    tc = host.toolchains[toolchain_name]

    # Resolve effective cargo_target
    effective_target = cargo_target or tc.cargo_target
    if not effective_target:
        raise ValueError(
            f"No cargo_target specified for toolchain {toolchain_name!r}. "
            "Set it in devices.toml [build] target or toolchain cargo_target."
        )

    # Build the uppercase env key prefix: aarch64-linux-android → AARCH64_LINUX_ANDROID
    triple_upper = effective_target.replace("-", "_").upper()

    result: dict[str, str] = {}

    if tc.ndk_home and tc.host_tag and tc.api_level:
        # Android NDK case
        toolchain_bin = (
            Path(tc.ndk_home)
            / "toolchains"
            / "llvm"
            / "prebuilt"
            / tc.host_tag
            / "bin"
        )
        clang = toolchain_bin / f"{effective_target}{tc.api_level}-clang"

        if not clang.exists():
            raise ToolchainNotFoundError(
                f"Android NDK clang not found. expected: {clang}"
            )

        clangxx = toolchain_bin / f"{effective_target}{tc.api_level}-clang++"
        llvm_ar = toolchain_bin / "llvm-ar"

        result[f"CC_{effective_target.replace('-', '_')}"] = str(clang)
        result[f"CXX_{effective_target.replace('-', '_')}"] = str(clangxx)
        result[f"AR_{effective_target.replace('-', '_')}"] = str(llvm_ar)
        result[f"CARGO_TARGET_{triple_upper}_LINKER"] = str(clang)

    elif tc.bin_prefix:
        # Cross-SDK case (generic cross toolchain)
        cc = tc.bin_prefix + "gcc"
        result[f"CC_{effective_target.replace('-', '_')}"] = cc
        result[f"CXX_{effective_target.replace('-', '_')}"] = tc.bin_prefix + "g++"
        result[f"AR_{effective_target.replace('-', '_')}"] = tc.bin_prefix + "ar"
        result[f"CARGO_TARGET_{triple_upper}_LINKER"] = cc

    # Merge extra env from toolchain config
    result.update(tc.env)

    return result


def build_binary(
    device_config: DeviceConfig,
    binary_name: str,
    project_root: str | Path,
    extra_bins: list[str] | None = None,
    dry_run: bool = False,
) -> bool:
    """Build a binary for the given device target.

    Returns True on success, False on failure.

    Env resolution priority:
    1. device_config.build.toolchain (new path via hosts.toml)
    2. device_config.build.env_file  (legacy, emits DeprecationWarning once)
    3. raw os.environ
    """
    global _warned_env_file

    project_root = Path(project_root)

    # Start from a copy of the current environment
    env = dict(os.environ)

    use_zigbuild = bool(device_config.build.zig_target)

    if use_zigbuild:
        # zigbuild handles linker/CC itself via zig — skip toolchain env composition.
        pass
    elif device_config.build.toolchain:
        # New path: compose env from hosts.toml
        try:
            hosts_cfg = load_hosts_config()
        except FileNotFoundError as exc:
            print(
                f"Error: {exc}\n"
                "Create hosts.toml from hosts.toml.example or run:\n"
                "  python scripts/device_registry.py bootstrap-host",
                flush=True,
            )
            return False

        try:
            host = detect_current_host(hosts_cfg)
            toolchain_env = _compose_toolchain_env(
                host,
                device_config.build.toolchain,
                device_config.build.target or None,
            )
        except (KeyError, ValueError, ToolchainNotFoundError, RuntimeError) as exc:
            print(f"Error resolving toolchain: {exc}")
            return False

        env.update(toolchain_env)

    elif device_config.build.env_file:
        # Legacy path
        if not _warned_env_file:
            warnings.warn(
                "BuildConfig.env_file is deprecated. "
                "Migrate to toolchain = \"android-ndk\" in devices.toml and create hosts.toml. "
                "See hosts.toml.example or run: python scripts/device_registry.py bootstrap-host",
                DeprecationWarning,
                stacklevel=2,
            )
            _warned_env_file = True

        env_file = project_root / device_config.build.env_file
        if env_file.exists():
            env = _source_env_file(env_file, env)
        else:
            print(f"Warning: env_file '{env_file}' not found, skipping")

    # Build cargo command
    if use_zigbuild:
        cmd_parts = ["cargo", "zigbuild", "--release"]
    else:
        cmd_parts = ["cargo", "build", "--release"]

    if device_config.build.features:
        cmd_parts.extend(["--features", ",".join(device_config.build.features)])

    if not device_config.build.default_features:
        cmd_parts.append("--no-default-features")

    if use_zigbuild:
        cmd_parts.extend(["--target", device_config.build.zig_target])
    elif device_config.build.target:
        cmd_parts.extend(["--target", device_config.build.target])

    cmd_parts.extend(["--bin", binary_name])

    if extra_bins:
        for b in extra_bins:
            cmd_parts.extend(["--bin", b])

    cmd = " ".join(cmd_parts)
    print(f"  Build: {cmd}")

    if dry_run:
        print("  (dry-run: skipping build)")
        return True

    r = subprocess.run(
        cmd, shell=True, cwd=str(project_root), env=env
    )
    return r.returncode == 0


def get_local_binary_path(
    device_config: DeviceConfig,
    binary_name: str,
    project_root: str | Path,
) -> Path:
    """Get the local path to a built binary."""
    project_root = Path(project_root)
    if device_config.build.binary_dir:
        return project_root / device_config.build.binary_dir / binary_name
    # Fallback: construct from target
    if device_config.build.target:
        return project_root / "target" / device_config.build.target / "release" / binary_name
    return project_root / "target" / "release" / binary_name


def _source_env_file(env_file: Path, base_env: dict) -> dict:
    """Source a shell env file and capture the resulting environment."""
    cmd = f"set -a && source {env_file} && env"
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            env=base_env, executable="/bin/bash"
        )
        if r.returncode == 0:
            new_env = dict(base_env)
            for line in r.stdout.splitlines():
                if "=" in line:
                    key, _, val = line.partition("=")
                    new_env[key] = val
            return new_env
    except Exception:
        pass
    return base_env
