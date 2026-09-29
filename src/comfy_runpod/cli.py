"""comfy — drive ComfyUI on a rented Runpod GPU."""

import argparse
import os
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

from .comfyui import ComfyUI, ComfyUIError
from .config import ConfigError, Infra, clear_state, load_infra, read_state, write_state
from .provision import provision
from .runpod_api import Client, RunpodError
from .tunnel import Tunnel, TunnelError

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


def _bring_up(client: Client, infra: Infra) -> str:
    """Pick a GPU with capacity, create the pod, and wait until SSH is reachable.

    Extracted out of `cmd_up` so `run` (Task 8) can reuse the exact same
    bring-up sequence. Returns the pod id.
    """
    choice, _ = client.pick_gpu(infra.gpus, infra.datacenter)
    pod = client.create_pod(
        name="comfy-up",
        template_id=choice.template,  # tied to the GPU's CUDA line — must match
        datacenter=infra.datacenter,
        volume_id=infra.volume_id,
        gpu_id=choice.id,
    )
    pod_id = str(pod["id"])

    deadline = time.time() + _SSH_READY_TIMEOUT
    while time.time() < deadline:
        try:
            client.ssh_target(pod_id)
            return pod_id
        except RunpodError:
            time.sleep(10)
    raise RunpodError(f"pod {pod_id} never exposed SSH within {_SSH_READY_TIMEOUT}s")


def cmd_up(args) -> int:
    infra = load_infra(INFRA_PATH)
    client = _client()

    # Picked here (not just inside _bring_up) so the choice is available to
    # report and to persist in state; _bring_up repeats the pick internally
    # to stay self-contained for `run`'s sake. Runpod's catalog is not
    # volatile enough for the two calls to disagree within one invocation.
    choice, availability = client.pick_gpu(infra.gpus, infra.datacenter)
    print(f"GPU: {choice.id} (availability: {availability})")

    print("creating pod...")
    pod_id = _bring_up(client, infra)
    print(f"pod: {pod_id}")
    started = int(time.time())

    # Recorded now, before the tunnel and the ComfyUI wait can fail: a pod
    # that exists must always be findable by `down`, even if this command
    # dies before it would otherwise reach the write_state below.
    write_state({"pod_id": pod_id, "started": started, "gpu": choice.id})

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

    write_state(
        {
            "pod_id": pod_id,
            "started": started,
            "gpu": choice.id,
            "tunnel_pid": tun.pid,
        }
    )

    print(f"url: {tun.url}")
    print(
        f"ComfyUI {system.get('comfyui_version', '?')}"
        f" — torch {system.get('pytorch_version', '?')}"
    )
    return 0


handlers["up"] = cmd_up


def _kill_tunnel(pid: int | None) -> None:
    if not pid:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # already gone


def cmd_down(args) -> int:
    state = read_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("no pod is running")
        return 0

    client = _client()
    spend = client.pod_spend(pod_id)  # read before terminating
    client.terminate_pod(pod_id)
    _kill_tunnel(state.get("tunnel_pid"))
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
