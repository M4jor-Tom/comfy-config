import argparse
import os
import signal
import time

import pytest

import comfy_runpod.cli as cli
from comfy_runpod.comfyui import ComfyUIError
from comfy_runpod.config import ConfigError, GpuChoice, Infra, Run, read_state, write_state
from comfy_runpod.runpod_api import RunpodError
from comfy_runpod.tunnel import TunnelError
from comfy_runpod.workflow import WorkflowError

BLACKWELL = GpuChoice(id="NVIDIA RTX PRO 4500 Blackwell", template="wgd3p4n4o6")
ADA = GpuChoice(id="NVIDIA GeForce RTX 4090", template="cw3nka7d08")


def _infra():
    return Infra(datacenter="EU-RO-1", gpus=[BLACKWELL, ADA], volume_id="vol_abc")


class FakeCliClient:
    """Stands in for runpod_api.Client for cli.py handler tests. Same method
    names as the real Client, no network."""

    def __init__(self, *, pod_id="pod-1", availability="HIGH",
                 desired_status="RUNNING", spend=0.42,
                 pods=None, volumes=None):
        self.pod_id = pod_id
        self.availability = availability
        self.desired_status = desired_status
        self.spend = spend
        self.pods = pods if pods is not None else []
        self.volumes = volumes if volumes is not None else []
        self.calls: list[str] = []
        self.terminated: list[str] = []
        self.deleted_volumes: list[str] = []
        self.created_with: dict | None = None

    def pick_gpu(self, gpus, datacenter):
        self.calls.append("pick_gpu")
        return gpus[0], self.availability

    def create_pod(self, *, name, template_id, datacenter, volume_id, gpu_id=None, cpu=None):
        self.calls.append("create_pod")
        self.created_with = {
            "name": name, "template_id": template_id, "datacenter": datacenter,
            "volume_id": volume_id, "gpu_id": gpu_id, "cpu": cpu,
        }
        return {"id": self.pod_id}

    def ssh_target(self, pod_id):
        self.calls.append("ssh_target")
        return ("1.2.3.4", 40022, "root")

    def get_pod(self, pod_id):
        self.calls.append("get_pod")
        return {"id": pod_id, "desiredStatus": self.desired_status}

    def terminate_pod(self, pod_id):
        self.calls.append("terminate_pod")
        self.terminated.append(pod_id)

    def pod_spend(self, pod_id):
        self.calls.append("pod_spend")
        return self.spend

    def list_pods(self):
        self.calls.append("list_pods")
        return self.pods

    def list_volumes(self):
        self.calls.append("list_volumes")
        return self.volumes

    def delete_volume(self, volume_id):
        self.calls.append("delete_volume")
        self.deleted_volumes.append(volume_id)


class FakeTunnel:
    """Stands in for tunnel.Tunnel: no subprocess, no socket."""

    last_instance = None

    def __init__(self, host, port, user, local_port=8188):
        self.host, self.port, self.user, self.local_port = host, port, user, local_port
        self.entered = False
        self.pid = 54321
        FakeTunnel.last_instance = self

    def __enter__(self):
        self.entered = True
        return self

    @property
    def url(self):
        return f"http://127.0.0.1:{self.local_port}"

    def close(self):
        self.entered = False


class FakeComfy:
    """Stands in for comfyui.ComfyUI: no HTTP."""

    def __init__(self, base_url):
        self.base_url = base_url

    def wait_ready(self, timeout=900):
        return {"system": {"comfyui_version": "0.3.0", "pytorch_version": "2.4.0+cu124"}}


# --- _bring_up ---------------------------------------------------------


def test_bring_up_returns_pod_id_choice_and_availability_once_ssh_is_ready(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient(pod_id="pod-99", availability="HIGH")
    pod_id, choice, availability = cli._bring_up(client, _infra())
    assert pod_id == "pod-99"
    assert choice == BLACKWELL
    assert availability == "HIGH"
    assert "create_pod" in client.calls
    assert "ssh_target" in client.calls


def test_bring_up_uses_the_picked_gpus_own_matching_template(monkeypatch, tmp_path):
    """The template is tied to the GPU's CUDA line -- mixing them is the
    brief's documented most-common first-run failure."""
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient()
    client.pick_gpu = lambda gpus, dc: (ADA, "MEDIUM")  # force the fallback GPU
    cli._bring_up(client, _infra())
    assert client.created_with["gpu_id"] == ADA.id
    assert client.created_with["template_id"] == ADA.template


def test_bring_up_polls_ssh_target_until_it_stops_raising(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient(pod_id="pod-5")
    real_ssh_target = client.ssh_target
    attempts = {"n": 0}

    def flaky(pod_id):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise RunpodError("runtime is still null")
        return real_ssh_target(pod_id)

    client.ssh_target = flaky
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)

    pod_id, _, _ = cli._bring_up(client, _infra())
    assert pod_id == "pod-5"
    assert attempts["n"] == 3


def test_bring_up_gives_up_after_the_ssh_timeout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient()

    def never_ready(pod_id):
        raise RunpodError("runtime is still null")

    client.ssh_target = never_ready
    monkeypatch.setattr(cli, "_SSH_READY_TIMEOUT", 0)  # expires before the first check
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)

    with pytest.raises(RunpodError, match="never exposed SSH"):
        cli._bring_up(client, _infra())


def test_bring_up_persists_pod_id_before_the_ssh_wait_so_a_timeout_does_not_leak_it(
    monkeypatch, tmp_path
):
    """Critical fix (Finding 1): the first-ever boot of a new template is
    exactly the plausible case for this loop to exhaust. The pod already
    exists and bills by the time create_pod returns -- state must record it
    before the wait loop, not only if/when _bring_up successfully returns."""
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient(pod_id="pod-13")

    def never_ready(pod_id):
        raise RunpodError("runtime is still null")

    client.ssh_target = never_ready
    monkeypatch.setattr(cli, "_SSH_READY_TIMEOUT", 0)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)

    with pytest.raises(RunpodError, match="never exposed SSH"):
        cli._bring_up(client, _infra())

    state = read_state()
    assert state["pod_id"] == "pod-13"
    assert state["gpu"] == BLACKWELL.id
    assert "tunnel_pid" not in state  # never got anywhere near the tunnel


# --- cmd_up --------------------------------------------------------------


def test_cmd_up_creates_pod_with_matching_template_opens_tunnel_and_persists_state(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient(pod_id="pod-42", availability="HIGH")
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)
    monkeypatch.setattr(cli, "ComfyUI", FakeComfy)

    rc = cli.cmd_up(argparse.Namespace())

    assert rc == 0
    assert client.created_with["template_id"] == BLACKWELL.template
    assert client.created_with["gpu_id"] == BLACKWELL.id
    assert client.created_with["volume_id"] == "vol_abc"
    assert FakeTunnel.last_instance.entered is True

    state = read_state()
    assert state["pod_id"] == "pod-42"
    assert state["gpu"] == BLACKWELL.id
    assert state["tunnel_pid"] == 54321
    assert state["tunnel_port"] == 8188
    assert isinstance(state["started"], int)

    out = capsys.readouterr().out
    assert "pod-42" in out
    assert "http://127.0.0.1:8188" in out
    assert "0.3.0" in out          # ComfyUI version
    assert "2.4.0+cu124" in out    # torch build
    assert "HIGH" in out           # availability


def test_cmd_up_calls_pick_gpu_exactly_once(monkeypatch, tmp_path):
    """Important fix (Finding 2): GPU stock genuinely moves within hours
    (observed live), so calling pick_gpu twice in one `up` risked printing
    and storing a GPU different from the one actually provisioned.
    _bring_up must be the only caller."""
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient()
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)
    monkeypatch.setattr(cli, "ComfyUI", FakeComfy)

    cli.cmd_up(argparse.Namespace())

    assert client.calls.count("pick_gpu") == 1


def test_cmd_up_falls_through_to_the_second_gpus_own_template(monkeypatch, tmp_path):
    """Regression guard for the brief's warning: the template must follow
    whichever GPU pick_gpu actually chose, not always the first configured one."""
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient(pod_id="pod-7")
    client.pick_gpu = lambda gpus, dc: (gpus[1], "MEDIUM")  # falls through to the 4090
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)
    monkeypatch.setattr(cli, "ComfyUI", FakeComfy)

    cli.cmd_up(argparse.Namespace())

    assert client.created_with["gpu_id"] == ADA.id
    assert client.created_with["template_id"] == ADA.template


def test_cmd_up_never_requests_public_ports_beyond_ssh(monkeypatch, tmp_path):
    """create_pod itself pins ports=['22/tcp'] (tested in test_runpod_api.py);
    this just confirms cmd_up doesn't pass anything that could override it."""
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient()
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)
    monkeypatch.setattr(cli, "ComfyUI", FakeComfy)

    cli.cmd_up(argparse.Namespace())

    assert set(client.created_with) == {
        "name", "template_id", "datacenter", "volume_id", "gpu_id", "cpu"
    }
    assert client.created_with["cpu"] is None


def test_cmd_up_persists_pod_id_before_the_tunnel_so_a_later_failure_does_not_leak_it(
    monkeypatch, tmp_path
):
    """If the tunnel fails to establish (ssh-readiness already succeeded
    inside _bring_up, which is what wrote this), `down` must still be able
    to find and terminate the pod. Complements the Finding-1 test above,
    which covers the earlier failure point (the ssh-wait loop itself)."""
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient(pod_id="pod-9")
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "_client", lambda: client)

    class ExplodingTunnel(FakeTunnel):
        def __enter__(self):
            raise TunnelError("port never opened")

    monkeypatch.setattr(cli, "Tunnel", ExplodingTunnel)
    monkeypatch.setattr(cli, "ComfyUI", FakeComfy)

    with pytest.raises(TunnelError):
        cli.cmd_up(argparse.Namespace())

    state = read_state()
    assert state["pod_id"] == "pod-9"   # recorded despite the later failure
    assert "tunnel_pid" not in state    # never got that far


# --- cmd_down --------------------------------------------------------------


def test_pid_cmdline_reads_the_real_proc_filesystem():
    """Sanity check against the actual OS, not a mock: confirms /proc really
    does hold our own argv (this test process's), null-byte-joined into a
    readable string. Finding 4's whole safety check depends on this working
    for real, not just against a fake in the tests below."""
    cmdline = cli._pid_cmdline(os.getpid())
    assert cmdline
    assert "python" in cmdline.lower() or "pytest" in cmdline.lower()


def test_pid_cmdline_raises_file_not_found_for_a_pid_that_does_not_exist():
    with pytest.raises(FileNotFoundError):
        cli._pid_cmdline(2**30)  # astronomically unlikely to be a real pid


def test_cmd_down_with_no_state_reports_and_exits_zero_without_calling_the_api(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient()
    monkeypatch.setattr(cli, "_client", lambda: client)

    rc = cli.cmd_down(argparse.Namespace())

    assert rc == 0
    assert client.calls == []
    out = capsys.readouterr().out.lower()
    assert "no pod" in out


def test_cmd_down_reads_spend_before_terminating_kills_tunnel_and_clears_state(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    write_state({
        "pod_id": "pod-1", "started": 1000, "gpu": "x",
        "tunnel_pid": 12345, "tunnel_port": 8188,
    })
    client = FakeCliClient(spend=1.23)
    monkeypatch.setattr(cli, "_client", lambda: client)
    # A live tunnel's /proc/<pid>/cmdline: ssh with the recorded -L spec.
    monkeypatch.setattr(
        cli, "_pid_cmdline",
        lambda pid: "ssh -N -T -L 8188:127.0.0.1:8188 root@1.2.3.4",
    )
    killed = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    rc = cli.cmd_down(argparse.Namespace())

    assert rc == 0
    assert client.calls.index("pod_spend") < client.calls.index("terminate_pod")
    assert client.terminated == ["pod-1"]
    assert killed == [(12345, signal.SIGTERM)]
    assert read_state() == {}
    assert "1.23" in capsys.readouterr().out


def test_cmd_down_tolerates_a_tunnel_pid_that_is_already_gone(monkeypatch, tmp_path):
    """/proc/<pid> no longer exists at all -- the ordinary, expected way a
    tunnel is already gone (ssh exited on its own, or a prior `down` already
    reaped it). Silent: no warning, no signal, no error."""
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-1", "started": 1000, "tunnel_pid": 99999, "tunnel_port": 8188})
    client = FakeCliClient()
    monkeypatch.setattr(cli, "_client", lambda: client)

    def gone(pid):
        raise FileNotFoundError

    monkeypatch.setattr(cli, "_pid_cmdline", gone)
    killed = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    assert cli.cmd_down(argparse.Namespace()) == 0  # must not raise
    assert read_state() == {}
    assert killed == []  # never reached os.kill -- nothing to signal


def test_cmd_down_tolerates_the_tunnel_dying_between_the_check_and_the_signal(
    monkeypatch, tmp_path
):
    """A narrower race than 'already gone': /proc/<pid>/cmdline still
    resolves (and matches our ssh tunnel) but the process has exited by the
    time we actually send the signal."""
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-1", "started": 1000, "tunnel_pid": 5555, "tunnel_port": 8188})
    client = FakeCliClient()
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(
        cli, "_pid_cmdline",
        lambda pid: "ssh -N -T -L 8188:127.0.0.1:8188 root@1.2.3.4",
    )

    def raise_lookup(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(cli.os, "kill", raise_lookup)

    assert cli.cmd_down(argparse.Namespace()) == 0  # must not raise
    assert read_state() == {}


def test_cmd_down_refuses_to_signal_a_pid_that_is_no_longer_our_tunnel(
    monkeypatch, tmp_path, capsys
):
    """Important fix (Finding 4): pids get recycled by the OS. If
    /proc/<pid>/cmdline no longer looks like the ssh -L forward we recorded,
    `down` must not signal it -- it could be an unrelated process by now."""
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-1", "started": 1000, "tunnel_pid": 7777, "tunnel_port": 8188})
    client = FakeCliClient()
    monkeypatch.setattr(cli, "_client", lambda: client)
    # pid 7777 now belongs to some unrelated process.
    monkeypatch.setattr(cli, "_pid_cmdline", lambda pid: "/usr/bin/some-other-daemon --serve")
    killed = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    rc = cli.cmd_down(argparse.Namespace())

    assert rc == 0
    assert killed == []  # must NOT have signalled the unrelated process
    out = capsys.readouterr().out.lower()
    assert "7777" in out
    assert "not signalling" in out


def test_cmd_down_declines_to_signal_when_tunnel_port_was_never_recorded(
    monkeypatch, tmp_path
):
    """Defence in depth: with no recorded port to match against, identity
    cannot be confirmed, so the safe default is to skip the kill."""
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-1", "started": 1000, "tunnel_pid": 4242})  # no tunnel_port
    client = FakeCliClient()
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(
        cli, "_pid_cmdline",
        lambda pid: "ssh -N -T -L 8188:127.0.0.1:8188 root@1.2.3.4",
    )
    killed = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    assert cli.cmd_down(argparse.Namespace()) == 0
    assert killed == []


def test_cmd_down_kills_the_tunnel_anyway_when_proc_is_unavailable(
    monkeypatch, tmp_path, capsys
):
    """Fix round 2: on a platform with no /proc at all, the identity check
    cannot be performed -- silently refusing to kill would leak an orphaned
    ssh -N forever (the pre-identity-check code always killed unconditionally,
    so this must not regress that). Must warn plainly and kill anyway."""
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-1", "started": 1000, "tunnel_pid": 8888, "tunnel_port": 8188})
    client = FakeCliClient()
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "_proc_available", lambda: False)
    killed = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    rc = cli.cmd_down(argparse.Namespace())

    assert rc == 0
    assert killed == [(8888, signal.SIGTERM)]  # killed anyway, not skipped
    out = capsys.readouterr().out.lower()
    assert "8888" in out
    assert "no /proc" in out or "cannot be verified" in out


def test_cmd_down_reports_unknown_spend_rather_than_crashing(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-1", "started": 1000})
    client = FakeCliClient(spend=None)
    monkeypatch.setattr(cli, "_client", lambda: client)

    assert cli.cmd_down(argparse.Namespace()) == 0
    assert "unknown" in capsys.readouterr().out


# --- cmd_status --------------------------------------------------------------


def test_cmd_status_with_no_pod_reports_nothing_running(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = cli.cmd_status(argparse.Namespace())
    assert rc == 0
    assert "no pod" in capsys.readouterr().out.lower()


def test_cmd_status_with_pod_reports_id_uptime_live_status_and_spend(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-1", "started": int(time.time()) - 90, "gpu": "x"})
    client = FakeCliClient(desired_status="RUNNING", spend=0.5)
    monkeypatch.setattr(cli, "_client", lambda: client)

    rc = cli.cmd_status(argparse.Namespace())

    assert rc == 0
    out = capsys.readouterr().out
    assert "pod-1" in out
    assert "RUNNING" in out
    assert "0.5" in out
    assert "get_pod" in client.calls
    assert "pod_spend" in client.calls


# --- cmd_teardown --------------------------------------------------------


def test_cmd_teardown_terminates_every_pod_unconditionally_and_clears_state(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-1", "started": 1000, "gpu": "x"})
    client = FakeCliClient(pods=[
        {"id": "pod-1", "name": "comfy-up"},
        {"id": "pod-2", "name": "comfy-run"},
    ])
    monkeypatch.setattr(cli, "_client", lambda: client)

    rc = cli.cmd_teardown(argparse.Namespace(delete_volume=False))

    assert rc == 0
    assert client.terminated == ["pod-1", "pod-2"]
    assert read_state() == {}  # cleared even though nothing else changed it
    out = capsys.readouterr().out
    assert "pod-1" in out and "comfy-up" in out
    assert "pod-2" in out and "comfy-run" in out


def test_cmd_teardown_with_nothing_running_says_so_plainly_and_exits_zero(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient(pods=[], volumes=[])
    monkeypatch.setattr(cli, "_client", lambda: client)

    rc = cli.cmd_teardown(argparse.Namespace(delete_volume=False))

    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "no pods running" in out
    assert "nothing is billing" in out
    assert client.deleted_volumes == []


def test_cmd_teardown_without_delete_volume_flag_reports_cost_and_never_deletes(
    monkeypatch, tmp_path, capsys
):
    """The default, no-flag path must never touch delete_volume -- deleting
    destroys every downloaded model irreversibly -- but must still surface
    what staying idle costs, since that's the whole point of the command."""
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient(pods=[], volumes=[
        {"id": "vol-1", "name": "comfy-models", "size": 75, "dataCenter": "EU-RO-1"},
    ])
    monkeypatch.setattr(cli, "_client", lambda: client)

    rc = cli.cmd_teardown(argparse.Namespace(delete_volume=False))

    assert rc == 0
    assert client.deleted_volumes == []
    assert "delete_volume" not in client.calls
    out = capsys.readouterr().out
    assert "vol-1" in out
    assert "5.25" in out  # 75 GB * $0.07/GB/month


def test_cmd_teardown_with_delete_volume_flag_deletes_every_volume(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    client = FakeCliClient(pods=[], volumes=[
        {"id": "vol-1", "size": 75},
        {"id": "vol-2", "size": 10},
    ])
    monkeypatch.setattr(cli, "_client", lambda: client)

    rc = cli.cmd_teardown(argparse.Namespace(delete_volume=True))

    assert rc == 0
    assert client.deleted_volumes == ["vol-1", "vol-2"]
    out = capsys.readouterr().out.lower()
    assert "nothing is billing" in out  # unambiguous: everything just got removed


def test_teardown_is_registered_in_the_dispatch_table():
    assert cli.handlers["teardown"] is cli.cmd_teardown


# --- main: centralised error handling (moved out of cmd_provision) ---------


def test_cmd_provision_no_longer_catches_errors_itself(monkeypatch):
    """The try/except moved to `main`; cmd_provision must let ConfigError and
    RunpodError propagate so main's wrapper is the only place that catches them."""
    def boom(path):
        raise ConfigError("comfy.yaml missing")

    monkeypatch.setattr(cli, "load_infra", boom)
    with pytest.raises(ConfigError):
        cli.cmd_provision(argparse.Namespace())


@pytest.mark.parametrize(
    "exc",
    [
        ConfigError("comfy.yaml missing"),
        RunpodError("API exploded"),
        TunnelError("port never opened"),
        ComfyUIError("ComfyUI never came up"),
    ],
)
def test_main_wraps_every_handler_error_as_a_clean_message(monkeypatch, capsys, exc):
    """main's try/except is the ONLY place any of these get caught now, for
    every handler -- not just cmd_provision, which used to catch its own."""
    def boom(args):
        raise exc

    monkeypatch.setitem(cli.handlers, "status", boom)
    rc = cli.main(["status"])

    assert rc == 1
    err = capsys.readouterr().err
    assert err.strip() == f"error: {exc}"


def test_main_returns_the_handlers_own_exit_code_on_success(monkeypatch):
    monkeypatch.setitem(cli.handlers, "status", lambda args: 0)
    assert cli.main(["status"]) == 0


def test_main_reports_unknown_commands_without_a_traceback(capsys):
    with pytest.raises(SystemExit):
        cli.main(["bogus-command"])
    # argparse itself rejects it (not one of the fixed subcommand choices)
    # before main's own "not implemented" branch is ever reached.
    assert "invalid choice" in capsys.readouterr().err


def test_up_down_status_are_registered_in_the_dispatch_table():
    """A handler that exists but is never assigned into `handlers` is silently
    unreachable from main() -- this is the mistake R5/R18 in progress.md warn
    about for every task that grows cli.py."""
    assert cli.handlers["up"] is cli.cmd_up
    assert cli.handlers["down"] is cli.cmd_down
    assert cli.handlers["status"] is cli.cmd_status


# --- cmd_run -----------------------------------------------------------------


class FakeRunComfy:
    """Stands in for comfyui.ComfyUI in cmd_run tests: enough control over
    queue/history/outputs_of/download to prove the batch loop's behaviour
    (incremental download, a named per-job failure) without any real HTTP."""

    last_instance = None

    def __init__(self, base_url):
        self.base_url = base_url
        self.uploaded = []
        self.queued = []       # [(graph, client_id), ...] -- successful queue() calls only
        self.downloaded = []   # [Path, ...]
        self.fail_on_job = None       # 1-based index at which queue() raises
        self.fail_message = "boom"
        FakeRunComfy.last_instance = self

    def wait_ready(self, timeout=900):
        return {"system": {}}

    def upload_image(self, path):
        self.uploaded.append(path)
        return f"server-{path.name}"

    def queue(self, graph, client_id):
        job_n = len(self.queued) + 1
        if self.fail_on_job == job_n:
            raise ComfyUIError(self.fail_message)
        self.queued.append((graph, client_id))
        return f"prompt-{job_n}"

    def history(self, prompt_id):
        return {"outputs": {}}  # always "done" immediately -- no polling wait in tests

    def outputs_of(self, prompt_id):
        return [{"filename": f"{prompt_id}.png", "subfolder": "", "type": "output"}]

    def download(self, entry, dest_dir):
        path = dest_dir / entry["filename"]
        self.downloaded.append(path)
        return path


def _run(**over):
    base = dict(mode="t2i", prompt="a fox in a forest", negative="blurry",
                count=3, size=(1024, 1024), seed=100, input=None, overrides={})
    base.update(over)
    return Run(**base)


def _patch_workflow(monkeypatch):
    """cmd_run's own orchestration is under test here, not workflow.py's
    internals (already covered exhaustively in test_workflow.py) -- so
    load/apply_run are faked to avoid needing real graph files under a
    chdir'd tmp_path, while seeds_for is left real (pure, no I/O) so
    per-job seed distinctness is proven against the actual implementation."""
    monkeypatch.setattr(cli.workflow, "load", lambda path: {"graph": True})
    monkeypatch.setattr(
        cli.workflow, "apply_run",
        lambda graph, run, seed, uploaded_name=None: {"seed": seed, "uploaded": uploaded_name},
    )


def test_cmd_run_rejects_invalid_run_file_before_any_pod_work(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("mode: t2x\nprompt: x\ncount: 1\n")
    client = FakeCliClient()
    monkeypatch.setattr(cli, "_client", lambda: client)

    with pytest.raises(ConfigError):
        cli.cmd_run(argparse.Namespace(run_file=str(bad), keep=True))

    assert client.calls == []       # _client() was never even reached
    assert read_state() == {}       # no pod ever tracked


def test_cmd_run_reuses_an_existing_pod_without_bringing_up_a_new_one(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-live", "tunnel_pid": 111, "tunnel_port": 8188})
    _patch_workflow(monkeypatch)
    FakeTunnel.last_instance = None
    client = FakeCliClient(pod_id="pod-live")
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "load_run", lambda path: _run(count=1))
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "ComfyUI", FakeRunComfy)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)

    rc = cli.cmd_run(argparse.Namespace(run_file="run.yaml", keep=True))

    assert rc == 0
    assert "create_pod" not in client.calls
    assert "pick_gpu" not in client.calls
    assert FakeTunnel.last_instance is None   # reused state's tunnel -- never opened a new one


def test_cmd_run_opens_a_fresh_tunnel_when_none_is_recorded(monkeypatch, tmp_path):
    """A pod with no tunnel yet in state (e.g. `_bring_up` just created it)
    must still get one -- the reuse path is only for a tunnel that is
    actually recorded."""
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-live"})  # pod exists, no tunnel_port
    _patch_workflow(monkeypatch)
    FakeTunnel.last_instance = None
    client = FakeCliClient(pod_id="pod-live")
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "load_run", lambda path: _run(count=1))
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "ComfyUI", FakeRunComfy)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)

    rc = cli.cmd_run(argparse.Namespace(run_file="run.yaml", keep=True))

    assert rc == 0
    assert FakeTunnel.last_instance is not None
    assert FakeTunnel.last_instance.entered is True
    # persisted so a later `comfy down` can find and close it, like `comfy up` does
    assert read_state()["tunnel_pid"] == FakeTunnel.last_instance.pid
    assert read_state()["tunnel_port"] == FakeTunnel.last_instance.local_port


def test_cmd_run_applies_a_distinct_seed_to_each_job_in_the_batch(monkeypatch, tmp_path):
    """Regression guard: a prior bug applied no per-image seed at all, so an
    entire batch came back as identical images."""
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-live", "tunnel_port": 8188})
    _patch_workflow(monkeypatch)
    client = FakeCliClient(pod_id="pod-live")
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "load_run", lambda path: _run(count=3, seed=100))
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "ComfyUI", FakeRunComfy)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)

    cli.cmd_run(argparse.Namespace(run_file="run.yaml", keep=True))

    seeds_used = [g["seed"] for g, _cid in FakeRunComfy.last_instance.queued]
    assert seeds_used == [100, 101, 102]   # workflow.seeds_for's real fixed-seed increment
    assert len(set(seeds_used)) == 3       # -- and genuinely distinct


def test_cmd_run_uploads_input_once_and_threads_the_server_name_into_every_job(
    monkeypatch, tmp_path
):
    """The server-reported name (which can differ from the local filename)
    must reach every job, not the local filename -- and the upload itself
    happens once per batch, not once per job."""
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-live", "tunnel_port": 8188})
    _patch_workflow(monkeypatch)
    img = tmp_path / "cat.png"
    img.write_bytes(b"x")
    client = FakeCliClient(pod_id="pod-live")
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "load_run", lambda path: _run(mode="i2v", count=2, input=img))
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "ComfyUI", FakeRunComfy)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)

    cli.cmd_run(argparse.Namespace(run_file="run.yaml", keep=True))

    comfy = FakeRunComfy.last_instance
    assert comfy.uploaded == [img]
    uploaded_names = [g["uploaded"] for g, _cid in comfy.queued]
    assert uploaded_names == ["server-cat.png", "server-cat.png"]


def test_cmd_run_keeps_completed_downloads_and_names_the_failed_job(monkeypatch, tmp_path):
    """The core batch-loop requirement: job i failing must not lose jobs
    1..i-1's already-downloaded output, and the error must say which job
    failed and why."""
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-live", "tunnel_port": 8188})
    _patch_workflow(monkeypatch)
    client = FakeCliClient(pod_id="pod-live")
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "load_run", lambda path: _run(count=3, seed=100))
    monkeypatch.setattr(cli, "_client", lambda: client)

    def make_comfy(base_url):
        c = FakeRunComfy(base_url)
        c.fail_on_job = 2
        c.fail_message = "GPU out of memory"
        return c

    monkeypatch.setattr(cli, "ComfyUI", make_comfy)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)

    with pytest.raises(ComfyUIError, match=r"job 2/3.*GPU out of memory"):
        cli.cmd_run(argparse.Namespace(run_file="run.yaml", keep=True))

    comfy = FakeRunComfy.last_instance
    assert len(comfy.queued) == 1       # job 2 never got recorded as queued
    assert len(comfy.downloaded) == 1   # -- but job 1's output was already downloaded


def test_cmd_run_converts_a_workflow_error_into_a_clean_message(monkeypatch, tmp_path):
    """WorkflowError isn't one of main()'s caught types -- cmd_run must not
    let it escape uncaught, or the user sees a raw traceback instead of
    `error: ...`."""
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-live", "tunnel_port": 8188})
    monkeypatch.setattr(cli.workflow, "load", lambda path: {"graph": True})

    def boom(graph, run, seed, uploaded_name=None):
        raise WorkflowError("workflow has no node titled 'positive'")

    monkeypatch.setattr(cli.workflow, "apply_run", boom)
    client = FakeCliClient(pod_id="pod-live")
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "load_run", lambda path: _run(count=1))
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "ComfyUI", FakeRunComfy)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)

    with pytest.raises(ComfyUIError, match="no node titled"):
        cli.cmd_run(argparse.Namespace(run_file="run.yaml", keep=True))


def test_cmd_run_terminates_the_pod_by_default_and_clears_state(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-live", "tunnel_port": 8188, "tunnel_pid": 999})
    _patch_workflow(monkeypatch)
    client = FakeCliClient(pod_id="pod-live", spend=0.11)
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "load_run", lambda path: _run(count=1))
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "ComfyUI", FakeRunComfy)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)
    monkeypatch.setattr(cli, "_kill_tunnel", lambda pid, port: None)

    rc = cli.cmd_run(argparse.Namespace(run_file="run.yaml", keep=False))

    assert rc == 0
    assert client.terminated == ["pod-live"]
    assert read_state() == {}
    assert "0.11" in capsys.readouterr().out


def test_cmd_run_keeps_the_pod_when_keep_is_passed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-live", "tunnel_port": 8188})
    _patch_workflow(monkeypatch)
    client = FakeCliClient(pod_id="pod-live")
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(cli, "load_run", lambda path: _run(count=1))
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "ComfyUI", FakeRunComfy)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)

    rc = cli.cmd_run(argparse.Namespace(run_file="run.yaml", keep=True))

    assert rc == 0
    assert client.terminated == []
    assert read_state()["pod_id"] == "pod-live"


def test_cmd_run_output_dir_is_dated_and_slugged_from_the_prompt(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    write_state({"pod_id": "pod-live", "tunnel_port": 8188})
    _patch_workflow(monkeypatch)
    client = FakeCliClient(pod_id="pod-live")
    monkeypatch.setattr(cli, "load_infra", lambda path: _infra())
    monkeypatch.setattr(
        cli, "load_run", lambda path: _run(count=1, prompt="A Red Fox!! In the Snow")
    )
    monkeypatch.setattr(cli, "_client", lambda: client)
    monkeypatch.setattr(cli, "ComfyUI", FakeRunComfy)
    monkeypatch.setattr(cli, "Tunnel", FakeTunnel)

    cli.cmd_run(argparse.Namespace(run_file="run.yaml", keep=True))

    [saved] = FakeRunComfy.last_instance.downloaded
    assert saved.parent.parent.name == "out"
    assert saved.parent.name.endswith("a-red-fox-in-the-snow")
    assert saved.parent.name.startswith(cli.date.today().isoformat())


def test_run_is_registered_in_the_dispatch_table():
    assert cli.handlers["run"] is cli.cmd_run
