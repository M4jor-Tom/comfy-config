import pytest

from comfy_runpod.tunnel import Tunnel


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
