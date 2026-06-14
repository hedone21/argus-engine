"""Device discovery — scan local system and ADB devices."""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass, field


@dataclass
class DiscoveredDevice:
    suggested_id: str
    name: str
    connection_type: str  # "local" or "adb"
    serial: str = ""
    arch: str = ""
    gpu: str = ""
    os_name: str = ""
    memory_gb: int = 0
    thermal_zone_types: list[str] = field(default_factory=list)

    def to_toml_dict(self) -> dict:
        """Convert to a dict structure matching devices.toml schema."""
        d: dict = {"name": self.name}

        conn: dict = {"type": self.connection_type}
        if self.serial:
            conn["serial"] = self.serial
        d["connection"] = conn

        build: dict = {}
        if self.connection_type == "adb":
            build["target"] = "aarch64-linux-android"
            build["toolchain"] = "android-ndk"
            build["binary_dir"] = "target/aarch64-linux-android/release"
        else:
            build["target"] = ""
            build["binary_dir"] = "target/release"
        d["build"] = build

        paths: dict = {}
        if self.connection_type == "local":
            paths["work_dir"] = "/tmp/argus"
            paths["model_dir"] = "models/llama3.2-1b"
        else:
            paths["work_dir"] = "/data/local/tmp"
            paths["model_dir"] = "/data/local/tmp/models/llama3.2-1b"
            paths["eval_dir"] = "/data/local/tmp/argus/eval"
            paths["lib_dir"] = "/data/local/tmp"
        d["paths"] = paths

        caps: dict = {}
        if self.arch:
            caps["arch"] = self.arch
        if self.gpu:
            caps["gpu"] = self.gpu
        if self.os_name:
            caps["os"] = self.os_name
        if self.memory_gb:
            caps["memory_gb"] = self.memory_gb
        d["capabilities"] = caps

        if self.connection_type == "adb" and self.thermal_zone_types:
            d["manager"] = {
                "transport": "unix",
                "thermal_zone_types": self.thermal_zone_types,
            }

        return d


def discover_local() -> DiscoveredDevice:
    """Discover local host capabilities."""
    import os
    import socket

    arch = platform.machine()
    hostname = socket.gethostname()
    memory_gb = 0

    try:
        with open("/proc/meminfo") as f:
            line = f.readline()
            m = re.search(r"(\d+)", line)
            if m:
                memory_gb = int(m.group(1)) // (1024 * 1024)
    except (OSError, ValueError):
        pass

    gpu = _detect_local_gpu()

    return DiscoveredDevice(
        suggested_id="host",
        name=hostname,
        connection_type="local",
        arch=arch,
        gpu=gpu or "none",
        memory_gb=memory_gb,
    )


def _detect_local_gpu() -> str:
    try:
        r = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                lower = line.lower()
                if "vga" in lower or "3d" in lower or "display" in lower:
                    if "nvidia" in lower:
                        return "nvidia"
                    elif "amd" in lower or "radeon" in lower:
                        return "amd"
                    elif "intel" in lower:
                        return "intel"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def discover_adb() -> list[DiscoveredDevice]:
    """Discover connected ADB devices."""
    try:
        r = subprocess.run(
            ["adb", "devices", "-l"], capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if r.returncode != 0:
        return []

    devices = []
    for line in r.stdout.strip().splitlines()[1:]:
        line = line.strip()
        if not line or "offline" in line:
            continue
        parts = line.split()
        if len(parts) < 2 or parts[1] != "device":
            continue
        serial = parts[0]
        dev = _probe_adb_device(serial)
        if dev:
            devices.append(dev)
    return devices


def _adb_getprop(serial: str, prop: str) -> str:
    try:
        r = subprocess.run(
            ["adb", "-s", serial, "shell", f"getprop {prop}"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _adb_shell(serial: str, cmd: str) -> str:
    try:
        r = subprocess.run(
            ["adb", "-s", serial, "shell", cmd],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _probe_adb_device(serial: str) -> DiscoveredDevice | None:
    model = _adb_getprop(serial, "ro.product.model")
    if not model:
        model = serial

    arch = _adb_shell(serial, "uname -m") or "aarch64"

    # Memory
    memory_gb = 0
    meminfo = _adb_shell(serial, "cat /proc/meminfo")
    if meminfo:
        m = re.search(r"MemTotal:\s+(\d+)", meminfo)
        if m:
            memory_gb = int(m.group(1)) // (1024 * 1024)

    # GPU detection
    gpu = _detect_adb_gpu(serial)

    # Thermal zones
    thermal_types = _discover_adb_thermal_zones(serial)

    suggested_id = re.sub(r"[^a-z0-9]", "_", model.lower()).strip("_")
    # Simplify common patterns
    suggested_id = re.sub(r"_+", "_", suggested_id)
    if len(suggested_id) > 20:
        suggested_id = suggested_id[:20].rstrip("_")

    return DiscoveredDevice(
        suggested_id=suggested_id,
        name=model,
        connection_type="adb",
        serial=serial,
        arch=arch,
        gpu=gpu,
        os_name="android",
        memory_gb=memory_gb,
        thermal_zone_types=thermal_types,
    )


def _detect_adb_gpu(serial: str) -> str:
    hardware = _adb_getprop(serial, "ro.hardware").lower()
    board = _adb_getprop(serial, "ro.board.platform").lower()

    if "qcom" in hardware or "qcom" in board or "sm" in board:
        return "adreno"
    elif "mali" in hardware:
        return "mali"
    elif "exynos" in board:
        return "mali"
    elif "mt" in board or "mediatek" in board:
        return "mali"

    # Fallback: check dumpsys
    gpu_info = _adb_shell(serial, "dumpsys SurfaceFlinger 2>/dev/null | head -5")
    if "adreno" in gpu_info.lower():
        return "adreno"
    elif "mali" in gpu_info.lower():
        return "mali"

    return "unknown"


def _discover_adb_thermal_zones(serial: str) -> list[str]:
    """Scan thermal zones on an ADB device and return CPU-related zone types."""
    output = _adb_shell(
        serial,
        "for z in /sys/class/thermal/thermal_zone*/type; do cat $z 2>/dev/null; done"
    )
    if not output:
        return []

    types = []
    for line in output.splitlines():
        t = line.strip()
        if t and "cpu" in t.lower():
            types.append(t)

    # Deduplicate, keep first occurrence
    seen = set()
    unique = []
    for t in types:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    # Return up to 3 representative types
    return unique[:3]


def discover_all() -> list[DiscoveredDevice]:
    """Run all discovery methods."""
    devices = [discover_local()]
    devices.extend(discover_adb())
    return devices
