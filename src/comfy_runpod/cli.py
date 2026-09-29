"""comfy — drive ComfyUI on a rented Runpod GPU."""

import argparse
import os
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

from .comfyui import ComfyUI, ComfyUIError
from .config import (
    ConfigError,
    GpuChoice,
    Infra,
    clear_state,
    load_infra,
    read_state,
    write_state,
)
from .provision import provision
from .runpod_api import Client, RunpodError
from .tunnel import Tunnel, TunnelError, forward_spec

COMMANDS = ("provision", "up", "run", "down", "status")

handlers = {}

INFRA_PATH = Path("comfy.yaml")

# How long to wait for a freshly created pod to expose SSH (runtime.ports).
_SSH_READY_TIMEOUT = 900


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="comfy", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("provision", help="one-time: create volume and download models")
    sub.add_parser("up", help="start the GPU pod and open the tunnel")
    r = sub.add_parser("run", help="run a batch from a run file")
    r.add_argument("run_file")
    r.add_argument("--keep", action="store_true", help="do not terminate when done")
    sub.add_parser("down", help="terminate the pod and report spend")
    sub.add_parser("status", help="show pod state and real spend")
    return p


def _client() -> Client:
    return Client(os.environ.get("RUNPOD_API_KEY", ""))


def cmd_provision(args) -> int:
    infra = load_infra(INFRA_PATH)
    volume_id = provision(_client(), infra)
    print(f"\nvolume_id: {volume_id}\nAdd that to {INFRA_PATH} before `comfy up`.")
    return 0


handlers["provision"] = cmd_provision


def _bring_up(client: Client, infra: Infra) -> tuple[str, GpuChoice, str]:
    """Pick a GPU with capacity, create the pod, and wait until SSH is reachable.

    Returns (pod_id, the GpuChoice actually used, its availability) so a
    caller never has to re-pick and risk disagreeing with what was actually
    provisioned -- GPU stock genuinely moves within hours, so two separate
    pick_gpu calls in one invocation are not guaranteed to agree.

    Persists {pod_id, started, gpu} the instant the pod exists, before the
    ssh-wait loop below -- which can run for up to _SSH_READY_TIMEOUT and is
    exactly where a pod is most likely to get stuck on a first boot of a new
    template. Without this, a timeout here would raise with the pod already
    created, running, and billing, but completely untracked: `down` would
    report nothing running.

    Extracted out of `cmd_up` so `run` (Task 8) can reuse the exact same
    bring-up sequence, early persistence included.
    """
    choice, availability = client.pick_gpu(infra.gpus, infra.datacenter)
    pod = client.create_pod(
        name="comfy-up",
        template_id=choice.template,  # tied to the GPU's CUDA line — must match
        datacenter=infra.datacenter,
        volume_id=infra.volume_id,
        gpu_id=choice.id,
    )
    pod_id = str(pod["id"])
    write_state({"pod_id": pod_id, "started": int(time.time()), "gpu": choice.id})

    deadline = time.time() + _SSH_READY_TIMEOUT
    while time.time() < deadline:
        try:
            client.ssh_target(pod_id)
            return pod_id, choice, availability
        except RunpodError:
            time.sleep(10)
    raise RunpodError(f"pod {pod_id} never exposed SSH within {_SSH_READY_TIMEOUT}s")


def cmd_up(args) -> int:
    infra = load_infra(INFRA_PATH)
    client = _client()

    print("creating pod...")
    pod_id, choice, availability = _bring_up(client, infra)
    print(f"GPU: {choice.id} (availability: {availability})")
    print(f"pod: {pod_id}")

    host, port, user = client.ssh_target(pod_id)
    print("opening tunnel...")
    tun = Tunnel(host, port, user)
    # Not a `with` block: the tunnel must outlive this command, so __exit__
    # (which would kill the ssh process) must never run here. `down` kills it
    # later via the pid recorded in state.
    tun.__enter__()

    print("waiting for ComfyUI...")
    stats = ComfyUI(tun.url).wait_ready()
    system = stats.get("system", {})

    # _bring_up already wrote pod_id/started/gpu; add the tunnel's identity
    # to that same record rather than reconstructing it from scratch.
    state = read_state()
    state["tunnel_pid"] = tun.pid
    state["tunnel_port"] = tun.local_port
    write_state(state)

    print(f"url: {tun.url}")
    print(
        f"ComfyUI {system.get('comfyui_version', '?')}"
        f" — torch {system.get('pytorch_version', '?')}"
    )
    return 0


handlers["up"] = cmd_up


def _proc_available() -> bool:
    """Whether /proc exists at all. False on a platform without it (this
    NixOS machine always has it, but the check must still degrade safely
    elsewhere) -- the ssh-identity check below cannot be performed there."""
    return Path("/proc").is_dir()


def _pid_cmdline(pid: int) -> str:
    """Space-joined argv of a running process, read from /proc. Only
    meaningful once _proc_available() is true. Raises FileNotFoundError if
    the pid no longer exists, or another OSError (e.g. PermissionError) if
    it exists but could not be read -- which happens when it now belongs to
    a different process entirely."""
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return raw.replace(b"\0", b" ").decode(errors="replace")


def _send_sigterm(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # gone between the caller's check and this signal


def _kill_tunnel(pid: int | None, port: int | None) -> None:
    """SIGTERM the tunnel, but only after confirming the pid is still our ssh
    process where that can be checked at all. A long `up` -> work -> `down`
    session can outlive the tunnel, and Linux recycles pids, so signalling a
    bare pid with no identity check risks hitting an unrelated process that
    happens to have inherited it.

    But without /proc there is no way to check at all, and silently
    refusing to kill in that case would leak the ssh process forever, which
    is worse than the small risk the check exists to guard against -- so
    that case warns plainly and kills anyway. /proc present but the pid
    gone is the ordinary, expected way a tunnel is already gone (stays
    silent). /proc present and the pid exists but doesn't look like our
    tunnel warns and skips.
    """
    if not pid:
        return

    if not _proc_available():
        print(f"tunnel pid {pid} cannot be verified on this platform (no /proc) — killing it anyway")
        _send_sigterm(pid)
        return

    try:
        cmdline = _pid_cmdline(pid)
    except FileNotFoundError:
        return  # already gone -- nothing to signal, nothing to warn about
    except OSError:
        print(f"tunnel pid {pid} could not be verified — not signalling it")
        return

    if not (port and "ssh" in cmdline and forward_spec(port) in cmdline):
        print(f"tunnel pid {pid} no longer looks like our ssh tunnel — not signalling it")
        return

    _send_sigterm(pid)


def cmd_down(args) -> int:
    state = read_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("no pod is running")
        return 0

    client = _client()
    spend = client.pod_spend(pod_id)  # read before terminating
    client.terminate_pod(pod_id)
    _kill_tunnel(state.get("tunnel_pid"), state.get("tunnel_port"))
    clear_state()

    spend_str = f"${spend:.2f}" if spend is not None else "unknown"
    print(f"terminated {pod_id} — spend: {spend_str}")
    return 0


handlers["down"] = cmd_down


def cmd_status(args) -> int:
    state = read_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("no pod is running")
        return 0

    client = _client()
    pod = client.get_pod(pod_id)
    spend = client.pod_spend(pod_id)
    uptime = timedelta(
        seconds=max(0, int(time.time()) - int(state.get("started", time.time())))
    )

    print(f"pod: {pod_id}")
    print(f"up for: {uptime}")
    print(f"status: {pod.get('desiredStatus', '?')}")
    print(f"spend: {f'${spend:.2f}' if spend is not None else 'unknown'}")
    return 0


handlers["status"] = cmd_status


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    handler = handlers.get(args.command)
    if handler is None:
        print(f"not implemented: {args.command}", file=sys.stderr)
        return 1
    try:
        return handler(args)
    except (ConfigError, RunpodError, TunnelError, ComfyUIError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
