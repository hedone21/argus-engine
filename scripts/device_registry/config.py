"""TOML configuration loader for device registry."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

_TOML_FILENAME = "devices.toml"
_HOSTS_TOML_FILENAME = "hosts.toml"


# ---------------------------------------------------------------------------
# Hosts / Toolchain config
# ---------------------------------------------------------------------------


@dataclass
class ToolchainConfig:
    """Per-toolchain build settings (NDK, cross-SDK, etc.)."""

    ndk_home: str = ""
    host_tag: str = ""
    api_level: int = 0
    sysroot: str = ""
    bin_prefix: str = ""
    cargo_target: str = ""
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Expand env vars and ~ in all string fields
        self.ndk_home = _expand(self.ndk_home)
        self.host_tag = _expand(self.host_tag)
        self.sysroot = _expand(self.sysroot)
        self.bin_prefix = _expand(self.bin_prefix)
        self.cargo_target = _expand(self.cargo_target)
        self.env = {k: _expand(v) for k, v in self.env.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolchainConfig:
        return cls(
            ndk_home=d.get("ndk_home", ""),
            host_tag=d.get("host_tag", ""),
            api_level=int(d.get("api_level", 0)),
            sysroot=d.get("sysroot", ""),
            bin_prefix=d.get("bin_prefix", ""),
            cargo_target=d.get("cargo_target", ""),
            env=dict(d.get("env", {})),
        )


@dataclass
class HostConfig:
    """Configuration for a single build host."""

    id: str
    uname_match: dict[str, str] = field(default_factory=dict)
    env_marker: str = ""
    toolchains: dict[str, ToolchainConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, host_id: str, d: dict[str, Any]) -> HostConfig:
        toolchains = {
            name: ToolchainConfig.from_dict(tc_dict)
            for name, tc_dict in d.get("toolchains", {}).items()
        }
        return cls(
            id=host_id,
            uname_match=dict(d.get("uname_match", {})),
            env_marker=d.get("env_marker", ""),
            toolchains=toolchains,
        )


@dataclass
class HostsConfig:
    """Root configuration loaded from hosts.toml."""

    schema_version: int
    default_host: str
    hosts: dict[str, HostConfig]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HostsConfig:
        hosts = {
            host_id: HostConfig.from_dict(host_id, host_data)
            for host_id, host_data in d.get("hosts", {}).items()
        }
        return cls(
            schema_version=int(d.get("schema_version", 1)),
            default_host=d.get("default_host", ""),
            hosts=hosts,
        )


def _expand(s: str) -> str:
    """Expand environment variables and ~ in a string."""
    return os.path.expanduser(os.path.expandvars(s))


def find_hosts_toml(start_dir: str | Path | None = None) -> Path:
    """Walk up from start_dir to find hosts.toml."""
    d = Path(start_dir) if start_dir else Path.cwd()
    while True:
        candidate = d / _HOSTS_TOML_FILENAME
        if candidate.is_file():
            return candidate
        parent = d.parent
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        f"{_HOSTS_TOML_FILENAME} not found in {start_dir or Path.cwd()} or any parent directory"
    )


def load_hosts_config(path: str | Path | None = None) -> HostsConfig:
    """Load and parse hosts.toml."""
    if path:
        p = Path(path)
    else:
        p = find_hosts_toml()
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return HostsConfig.from_dict(data)


def detect_current_host(
    hosts: HostsConfig,
    env: dict[str, str] | None = None,
) -> HostConfig:
    """Detect the current build host from hosts.toml.

    Priority:
    1. ARGUS_HOST env var override.
    2. env_marker match (truthy env var) + uname_match (all keys must match).
    3. Plain uname_match (all keys must match).
    4. hosts.default_host fallback.
    5. RuntimeError if all fail.
    """
    if env is None:
        env = dict(os.environ)

    # 1. Explicit override
    override = env.get("ARGUS_HOST", "")
    if override:
        if override in hosts.hosts:
            return hosts.hosts[override]
        available = list(hosts.hosts.keys())
        raise RuntimeError(
            f"ARGUS_HOST={override!r} not found in hosts.toml; "
            f"available hosts={available}"
        )

    uname = platform.uname()
    uname_fields = {
        "sysname": uname.system,
        "machine": uname.machine,
        "nodename": uname.node,
        "release": uname.release,
        "version": uname.version,
        "processor": uname.processor,
    }

    def _matches_uname(host: HostConfig) -> bool:
        for key, expected in host.uname_match.items():
            if uname_fields.get(key, "") != expected:
                return False
        return True

    # 2. env_marker-gated hosts (higher priority than plain match)
    for host in hosts.hosts.values():
        if host.env_marker and env.get(host.env_marker, ""):
            if _matches_uname(host):
                return host

    # 3. Plain uname match
    for host in hosts.hosts.values():
        if not host.env_marker and _matches_uname(host):
            return host

    # 4. default_host fallback
    if hosts.default_host and hosts.default_host in hosts.hosts:
        return hosts.hosts[hosts.default_host]

    available = list(hosts.hosts.keys())
    raise RuntimeError(
        f"No host matched. "
        f"detected uname={dict(uname_fields)!r} ; "
        f"available hosts={available} ; "
        f"set ARGUS_HOST=<id>"
    )


@dataclass
class ConnectionConfig:
    type: str  # "local", "adb", "ssh"
    serial: str = ""
    # SSH-specific fields
    host: str = ""
    user: str = ""
    port: int = 22
    identity_file: str = ""
    ssh_options: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConnectionConfig:
        return cls(
            type=d["type"],
            serial=d.get("serial", ""),
            host=d.get("host", ""),
            user=d.get("user", ""),
            port=int(d.get("port", 22)),
            identity_file=d.get("identity_file", ""),
            ssh_options=list(d.get("ssh_options", [])),
        )


@dataclass
class BuildConfig:
    target: str = ""  # empty = native build
    env_file: str = ""  # deprecated: use toolchain instead
    binary_dir: str = ""
    toolchain: str = ""  # toolchain name key in hosts.toml
    features: list[str] = field(default_factory=list)
    default_features: bool = True
    # zigbuild target (e.g. "aarch64-unknown-linux-gnu.2.31") — when set,
    # builder uses `cargo zigbuild` with this target and skips toolchain env composition.
    zig_target: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BuildConfig:
        return cls(
            target=d.get("target", ""),
            env_file=d.get("env_file", ""),
            binary_dir=d.get("binary_dir", ""),
            toolchain=d.get("toolchain", ""),
            features=list(d.get("features", [])),
            default_features=bool(d.get("default_features", True)),
            zig_target=d.get("zig_target", ""),
        )


@dataclass
class DevicePaths:
    work_dir: str = ""
    model_dir: str = ""
    eval_dir: str = ""
    lib_dir: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DevicePaths:
        return cls(
            work_dir=d.get("work_dir", ""),
            model_dir=d.get("model_dir", ""),
            eval_dir=d.get("eval_dir", ""),
            lib_dir=d.get("lib_dir", ""),
        )


@dataclass
class ManagerConfig:
    transport: str = ""
    thermal_zone_types: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ManagerConfig:
        return cls(
            transport=d.get("transport", ""),
            thermal_zone_types=d.get("thermal_zone_types", []),
        )


@dataclass
class DeviceConfig:
    id: str
    name: str
    connection: ConnectionConfig
    build: BuildConfig
    paths: DevicePaths
    capabilities: dict[str, Any] = field(default_factory=dict)
    manager: ManagerConfig | None = None

    @classmethod
    def from_dict(cls, device_id: str, d: dict[str, Any]) -> DeviceConfig:
        conn = ConnectionConfig.from_dict(d.get("connection", {"type": "local"}))
        build = BuildConfig.from_dict(d.get("build", {}))
        paths = DevicePaths.from_dict(d.get("paths", {}))
        caps = dict(d.get("capabilities", {}))
        mgr = ManagerConfig.from_dict(d["manager"]) if "manager" in d else None
        return cls(
            id=device_id,
            name=d.get("name", device_id),
            connection=conn,
            build=build,
            paths=paths,
            capabilities=caps,
            manager=mgr,
        )

    @property
    def is_local(self) -> bool:
        return self.connection.type == "local"

    @property
    def is_android(self) -> bool:
        return self.connection.type == "adb"

    def binary_remote_path(self, binary_name: str) -> str:
        return f"{self.paths.work_dir}/{binary_name}" if self.paths.work_dir else binary_name

    def ld_library_path_env(self) -> str:
        if self.paths.lib_dir:
            return f"LD_LIBRARY_PATH={self.paths.lib_dir}"
        return ""


def find_devices_toml(start_dir: str | Path | None = None) -> Path:
    """Walk up from start_dir to find devices.toml."""
    d = Path(start_dir) if start_dir else Path.cwd()
    while True:
        candidate = d / _TOML_FILENAME
        if candidate.is_file():
            return candidate
        parent = d.parent
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        f"{_TOML_FILENAME} not found in {start_dir or Path.cwd()} or any parent directory"
    )


def _load_toml(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path:
        p = Path(config_path)
    else:
        p = find_devices_toml()
    with open(p, "rb") as f:
        return tomllib.load(f)


def load_device_config(
    device_id: str, config_path: str | Path | None = None
) -> DeviceConfig:
    data = _load_toml(config_path)
    devices = data.get("devices", {})
    if device_id not in devices:
        available = ", ".join(sorted(devices.keys())) or "(none)"
        raise KeyError(f"Device '{device_id}' not found. Available: {available}")
    return DeviceConfig.from_dict(device_id, devices[device_id])


def load_all_devices(config_path: str | Path | None = None) -> dict[str, DeviceConfig]:
    data = _load_toml(config_path)
    devices = data.get("devices", {})
    return {
        dev_id: DeviceConfig.from_dict(dev_id, dev_data)
        for dev_id, dev_data in devices.items()
    }
