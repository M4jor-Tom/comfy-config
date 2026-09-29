import subprocess

import pytest

import comfy_runpod.tunnel as tunnel_mod
from comfy_runpod.tunnel import Tunnel, TunnelError


class _FakeOpenSocket:
    """Pretends the local port is already open, so _wait_for_local_port
    returns on its first check without ever touching a real socket."""

    def settimeout(self, t):
        pass

    def connect_ex(self, addr):
        return 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ssh_command_forwards_8188_and_disables_remote_command():
    cmd = Tunnel("1.2.3.4", 40022, "root").ssh_command()
    joined = " ".join(cmd)
    assert cmd[0] == "ssh"
    assert "-L" in cmd
    assert "8188:127.0.0.1:8188" in cmd
    assert "-N" in cmd            # no remote command, forwarding only
    assert "-T" in cmd            # no tty
    assert "-p" in cmd and "40022" in cmd
    assert "root@1.2.3.4" in joined
    assert "ExitOnForwardFailure=yes" in joined  # fail loudly, not silently


def test_custom_local_port_is_honoured():
    cmd = Tunnel("h", 22, "root", local_port=9999).ssh_command()
    assert "9999:127.0.0.1:8188" in cmd


def test_url_uses_loopback_not_localhost_name():
    t = Tunnel("h", 22, "root", local_port=9999)
    assert t.url == "http://127.0.0.1:9999"


def test_keepalive_is_set_so_a_long_video_render_does_not_drop():
    joined = " ".join(Tunnel("h", 22, "root").ssh_command())
    assert "ServerAliveInterval" in joined


def test_enter_wires_ssh_stderr_to_a_real_file_not_pipe_or_devnull(monkeypatch, tmp_path):
    """Finding 3: DEVNULL discarded the only diagnostic when ssh fails; a
    PIPE risks SIGPIPE once the tunnel is detached (its read end closes when
    the opener exits). A real, predictable file avoids both."""
    monkeypatch.chdir(tmp_path)
    captured = {}

    class FakeProc:
        returncode = None

        def __init__(self, cmd, **kwargs):
            captured["stderr"] = kwargs.get("stderr")
            captured["start_new_session"] = kwargs.get("start_new_session")

        def poll(self):
            return None  # still running

    monkeypatch.setattr(tunnel_mod.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(tunnel_mod.socket, "socket", lambda *a, **kw: _FakeOpenSocket())

    Tunnel("h", 22, "root").__enter__()

    assert captured["stderr"] not in (subprocess.PIPE, subprocess.DEVNULL)
    assert captured["stderr"] is not None
    assert captured["start_new_session"] is True


def test_wait_for_local_port_includes_the_logged_ssh_error_on_immediate_exit(
    monkeypatch, tmp_path
):
    """Finding 3: ssh returns 255 for nearly every failure (bad host key,
    auth failure, refused connection), so the exit code alone can't tell
    them apart -- the real ssh stderr must reach TunnelError."""
    monkeypatch.chdir(tmp_path)

    class FakeProc:
        returncode = 255

        def __init__(self, cmd, **kwargs):
            kwargs["stderr"].write(b"Permission denied (publickey).\n")
            kwargs["stderr"].flush()

        def poll(self):
            return 255  # already exited

    monkeypatch.setattr(tunnel_mod.subprocess, "Popen", FakeProc)

    with pytest.raises(TunnelError, match="Permission denied"):
        Tunnel("h", 22, "root").__enter__()
