"""SSH port-forward to ComfyUI. Port 8188 is never exposed publicly.

Runpod's proxy SSH cannot forward ports, so this requires the direct-TCP
endpoint, which in turn requires 22/tcp at pod creation.
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

# Where ssh's stderr goes. A real file, not a pipe: the tunnel can be
# detached (comfy up does this) so it outlives the process that opened it,
# and a pipe's read end closing when we exit would make a later write by
# ssh raise SIGPIPE and kill it -- the opposite of outliving the command.
# Fixed and predictable (not a random temp name) so a live tunnel's actual
# ssh errors are always in the same place to look, and so TunnelError can
# quote them instead of just an exit code.
LOG_PATH = Path(".comfy-tunnel.log")


class TunnelError(Exception):
    """The forward could not be established, or died."""


def forward_spec(local_port: int) -> str:
    """The -L argument's value, e.g. '8188:127.0.0.1:8188'. Shared with
    cli.py's pid-identity check in `down`, so both sides agree on what a
    live tunnel's ssh process should have on its command line."""
    return f"{local_port}:127.0.0.1:8188"


class Tunnel:
    def __init__(self, host: str, port: int, user: str, local_port: int = 8188) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.local_port = local_port
        self._proc: subprocess.Popen | None = None
        self._log_path = LOG_PATH

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    @property
    def pid(self) -> int | None:
        """The ssh process id, so callers who need the tunnel to outlive them
        (`comfy up`) can persist it and a later process can kill it by pid."""
        return self._proc.pid if self._proc else None

    def ssh_command(self) -> list[str]:
        return [
            "ssh",
            "-N",  # forward only, run no remote command
            "-T",  # no pseudo-terminal
            "-p", str(self.port),
            "-L", forward_spec(self.local_port),
            "-o", "ExitOnForwardFailure=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=8",
            f"{self.user}@{self.host}",
        ]

    def __enter__(self) -> Tunnel:
        log = open(self._log_path, "wb")  # truncate: no stale content from a past run
        try:
            self._proc = subprocess.Popen(
                self.ssh_command(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                # New session so the tunnel is not in our controlling terminal's
                # process group: closing the shell must not SIGHUP it.
                start_new_session=True,
            )
        finally:
            log.close()  # the child dup'd its own fd; our copy can close now
        self._wait_for_local_port()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def _read_log(self) -> str:
        try:
            text = self._log_path.read_text(errors="replace").strip()
        except OSError:
            return "(no log)"
        return text[-400:] if text else "(ssh wrote nothing to stderr)"

    def _wait_for_local_port(self, timeout: int = 60) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise TunnelError(
                    f"ssh exited immediately with code {self._proc.returncode}: "
                    f"{self._read_log()} (full log: {self._log_path})"
                )
            with socket.socket() as s:
                s.settimeout(2)
                if s.connect_ex(("127.0.0.1", self.local_port)) == 0:
                    return
            time.sleep(1)
        self.close()
        raise TunnelError(
            f"port {self.local_port} never opened. Is something already using "
            f"it? ssh log: {self._read_log()} (full log: {self._log_path})"
        )
