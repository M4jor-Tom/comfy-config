"""SSH port-forward to ComfyUI. Port 8188 is never exposed publicly.

Runpod's proxy SSH cannot forward ports, so this requires the direct-TCP
endpoint, which in turn requires 22/tcp at pod creation.
"""

from __future__ import annotations

import socket
import subprocess
import time


class TunnelError(Exception):
    """The forward could not be established, or died."""


class Tunnel:
    def __init__(self, host: str, port: int, user: str, local_port: int = 8188) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.local_port = local_port
        self._proc: subprocess.Popen | None = None

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
            "-L", f"{self.local_port}:127.0.0.1:8188",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=8",
            f"{self.user}@{self.host}",
        ]

    def __enter__(self) -> Tunnel:
        self._proc = subprocess.Popen(
            self.ssh_command(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            # DEVNULL, not PIPE: a caller may detach this process (comfy up
            # does) so it outlives us. A PIPE's read end closes when we exit,
            # and ssh writing to it after that gets SIGPIPE and dies — exactly
            # the opposite of outliving the command.
            stderr=subprocess.DEVNULL,
            # New session so the tunnel is not in our controlling terminal's
            # process group: closing the shell must not SIGHUP it.
            start_new_session=True,
        )
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

    def _wait_for_local_port(self, timeout: int = 60) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise TunnelError(
                    f"ssh exited immediately with code {self._proc.returncode}"
                )
            with socket.socket() as s:
                s.settimeout(2)
                if s.connect_ex(("127.0.0.1", self.local_port)) == 0:
                    return
            time.sleep(1)
        self.close()
        raise TunnelError(
            f"port {self.local_port} never opened. Is something already using it?"
        )
