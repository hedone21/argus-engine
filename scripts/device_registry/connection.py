"""Connection abstraction for local and remote device execution."""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .config import ConnectionConfig


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Connection(ABC):
    """Abstract connection to a device (local, adb, ssh)."""

    @abstractmethod
    def execute(
        self,
        command: str,
        timeout: int | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> CommandResult:
        ...

    @abstractmethod
    def push(self, local_path: str | Path, remote_path: str) -> None:
        ...

    @abstractmethod
    def pull(self, remote_path: str, local_path: str | Path) -> None:
        ...

    @abstractmethod
    def file_exists(self, remote_path: str) -> bool:
        ...

    @abstractmethod
    def mkdir(self, remote_path: str) -> None:
        ...

    @abstractmethod
    def chmod(self, remote_path: str, mode: str = "+x") -> None:
        ...


class LocalConnection(Connection):
    """Connection for local host execution."""

    def execute(
        self,
        command: str,
        timeout: int | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> CommandResult:
        import os

        env = dict(os.environ)
        if env_vars:
            env.update(env_vars)
        try:
            r = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return CommandResult(r.returncode, r.stdout, r.stderr)
        except subprocess.TimeoutExpired:
            return CommandResult(-1, "", f"Command timed out after {timeout}s")

    def push(self, local_path: str | Path, remote_path: str) -> None:
        src = Path(local_path)
        dst = Path(remote_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def pull(self, remote_path: str, local_path: str | Path) -> None:
        self.push(remote_path, str(local_path))

    def file_exists(self, remote_path: str) -> bool:
        return Path(remote_path).exists()

    def mkdir(self, remote_path: str) -> None:
        Path(remote_path).mkdir(parents=True, exist_ok=True)

    def chmod(self, remote_path: str, mode: str = "+x") -> None:
        subprocess.run(["chmod", mode, remote_path], check=True)


class AdbConnection(Connection):
    """Connection via Android Debug Bridge."""

    def __init__(self, serial: str = ""):
        self._serial = serial

    def _adb_prefix(self) -> list[str]:
        if self._serial:
            return ["adb", "-s", self._serial]
        return ["adb"]

    def execute(
        self,
        command: str,
        timeout: int | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> CommandResult:
        env_prefix = ""
        if env_vars:
            env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items()) + " "
        shell_cmd = f"{env_prefix}{command} 2>&1"
        cmd = self._adb_prefix() + ["shell", shell_cmd]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return CommandResult(r.returncode, r.stdout, r.stderr)
        except subprocess.TimeoutExpired:
            return CommandResult(-1, "", f"Command timed out after {timeout}s")

    def push(self, local_path: str | Path, remote_path: str) -> None:
        cmd = self._adb_prefix() + ["push", str(local_path), remote_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"adb push failed: {r.stderr}")

    def pull(self, remote_path: str, local_path: str | Path) -> None:
        cmd = self._adb_prefix() + ["pull", remote_path, str(local_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"adb pull failed: {r.stderr}")

    def file_exists(self, remote_path: str) -> bool:
        result = self.execute(f"ls {remote_path}")
        return result.ok

    def mkdir(self, remote_path: str) -> None:
        self.execute(f"mkdir -p {remote_path}")

    def chmod(self, remote_path: str, mode: str = "+x") -> None:
        self.execute(f"chmod {mode} {remote_path}")


class SshConnection(Connection):
    """SSH connection via openssh client (key-based auth only)."""

    def __init__(
        self,
        host: str,
        user: str,
        port: int = 22,
        identity_file: str = "",
        extra_options: list[str] | None = None,
    ) -> None:
        if not host:
            raise ValueError("SshConnection: 'host' is required")
        if not user:
            raise ValueError("SshConnection: 'user' is required")
        self._host = host
        self._user = user
        self._port = port
        self._identity_file = identity_file
        self._extra_options: list[str] = list(extra_options or [])

    def _ssh_base(self) -> list[str]:
        """Build the base ssh argument list (without remote command)."""
        args = ["ssh", "-p", str(self._port)]
        if self._identity_file:
            args.extend(["-i", self._identity_file])
        args.extend(self._extra_options)
        args.append(f"{self._user}@{self._host}")
        return args

    def _scp_base(self) -> list[str]:
        """Build the base scp argument list (without source/dest)."""
        args = ["scp", "-P", str(self._port)]
        if self._identity_file:
            args.extend(["-i", self._identity_file])
        args.extend(self._extra_options)
        return args

    def execute(
        self,
        command: str,
        timeout: int | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> CommandResult:
        env_prefix = ""
        if env_vars:
            env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items()) + " "
        full_cmd = self._ssh_base() + [f"{env_prefix}{command}"]
        try:
            r = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return CommandResult(r.returncode, r.stdout, r.stderr)
        except subprocess.TimeoutExpired:
            return CommandResult(-1, "", f"Command timed out after {timeout}s")

    def push(self, local_path: str | Path, remote_path: str) -> None:
        local = Path(local_path)
        cmd = self._scp_base()
        if local.is_dir():
            cmd.append("-r")
        cmd.append(str(local))
        cmd.append(f"{self._user}@{self._host}:{remote_path}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"scp push failed: {r.stderr}")

    def pull(self, remote_path: str, local_path: str | Path) -> None:
        cmd = self._scp_base()
        cmd.append(f"{self._user}@{self._host}:{remote_path}")
        cmd.append(str(local_path))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"scp pull failed: {r.stderr}")

    def file_exists(self, remote_path: str) -> bool:
        result = self.execute(f"test -e {remote_path}")
        return result.returncode == 0

    def mkdir(self, remote_path: str) -> None:
        self.execute(f"mkdir -p {remote_path}")

    def chmod(self, remote_path: str, mode: str = "+x") -> None:
        self.execute(f"chmod {mode} {remote_path}")


def create_connection(config: ConnectionConfig) -> Connection:
    """Factory: create a Connection from ConnectionConfig."""
    if config.type == "local":
        return LocalConnection()
    elif config.type == "adb":
        return AdbConnection(serial=config.serial)
    elif config.type == "ssh":
        return SshConnection(
            host=config.host,
            user=config.user,
            port=config.port,
            identity_file=config.identity_file,
            extra_options=config.ssh_options,
        )
    else:
        raise ValueError(f"Unknown connection type: {config.type}")
