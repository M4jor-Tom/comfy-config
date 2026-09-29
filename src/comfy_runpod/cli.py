"""comfy — drive ComfyUI on a rented Runpod GPU."""

import argparse
import os
import re
import secrets
import signal
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from . import workflow
from .comfyui import ComfyUI, ComfyUIError
from .config import (
    ConfigError,
    GpuChoice,
    Infra,
    clear_state,
    load_infra,
    load_run,
    merge_state,
    read_state,
    write_state,
)
from .provision import provision
from .runpod_api import Client, RunpodError
from .tunnel import Tunnel, TunnelError, forward_spec
from .workflow import WorkflowError

INFRA_PATH = Path("comfy.yaml")
WORKFLOW_DIR = Path("workflows")

# How long to wait for a freshly created pod to expose SSH (runtime.ports).
_SSH_READY_TIMEOUT = 900

# Runpod network-volume pricing: $0.07/GB/month for the first 1 TB.
_VOLUME_GB_MONTHLY_USD = 0.07


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
    t = sub.add_parser("teardown", help="stop all billing: terminate pods, report volumes")
    t.add_argument(
        "--delete-volume",
        action="store_true",
        help="also delete every volume — irreversible, destroys downloaded models",
    )
    return p


def _client() -> Client:
    return Client(os.environ.get("RUNPOD_API_KEY", ""))


def cmd_provision(args) -> int:
    infra = load_infra(INFRA_PATH)
    volume_id = provision(_client(), infra)
    print(f"\nvolume_id: {volume_id}\nAdd that to {INFRA_PATH} before `comfy up`.")
    return 0


def _bring_up(
    client: Client, infra: Infra
) -> tuple[str, GpuChoice, str, tuple[str, int, str]]:
    """Pick a GPU with capacity, create the pod, and wait until SSH is reachable.

    Returns (pod_id, the GpuChoice actually used, its availability, its ssh
    target) so a caller never has to re-pick a GPU -- risking disagreement
    with what was actually provisioned, since GPU stock genuinely moves
    within hours -- or re-fetch the ssh target `wait_for_ssh` already found.

    Persists {pod_id, started, gpu} the instant the pod exists, before the
    ssh wait below -- which can run for up to _SSH_READY_TIMEOUT and is
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

    ssh = client.wait_for_ssh(pod_id, timeout=_SSH_READY_TIMEOUT)
    return pod_id, choice, availability, ssh


def cmd_up(args) -> int:
    infra = load_infra(INFRA_PATH)
    client = _client()

    print("creating pod...")
    pod_id, choice, availability, ssh = _bring_up(client, infra)
    print(f"GPU: {choice.id} (availability: {availability})")
    print(f"pod: {pod_id}")

    print("opening tunnel...")
    tun = Tunnel(*ssh)
    # Not a `with` block: the tunnel must outlive this command, so __exit__
    # (which would kill the ssh process) must never run here. `down` kills it
    # later via the pid recorded in state.
    tun.__enter__()

    print("waiting for ComfyUI...")
    stats = ComfyUI(tun.url).wait_ready()
    system = stats.get("system", {})

    # _bring_up already wrote pod_id/started/gpu; add the tunnel's identity
    # to that same record rather than reconstructing it from scratch.
    merge_state(tunnel_pid=tun.pid, tunnel_port=tun.local_port)

    print(f"url: {tun.url}")
    print(
        f"ComfyUI {system.get('comfyui_version', '?')}"
        f" — torch {system.get('pytorch_version', '?')}"
    )
    return 0


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


def _fmt_spend(spend: float | None) -> str:
    return f"${spend:.2f}" if spend is not None else "unknown"


def _terminate(client: Client, pod_id: str, state: dict, tun: Tunnel | None = None) -> float | None:
    """Read spend, terminate the pod, and close its tunnel. Shared by
    `cmd_down` and `cmd_run`'s default (non---keep) teardown so the
    spend-must-be-read-before-terminate ordering only has to be right once.

    `tun` is the live Tunnel object when the caller opened it itself in this
    same process (it can just close() the handle it's already holding);
    passing None means "identify and signal whatever tunnel state records"
    via `_kill_tunnel`, which is the only option when the caller merely
    reused a tunnel opened by someone else.
    """
    spend = client.pod_spend(pod_id)  # read before terminating
    client.terminate_pod(pod_id)
    if tun is not None:
        tun.close()
    else:
        _kill_tunnel(state.get("tunnel_pid"), state.get("tunnel_port"))
    clear_state()
    return spend


def _running_pod_id(state: dict) -> str | None:
    """The pod_id recorded in state, or None -- printing "no pod is
    running" as a side effect when there isn't one, since `down` and
    `status` both want exactly that message for exactly that condition."""
    pod_id = state.get("pod_id")
    if not pod_id:
        print("no pod is running")
    return pod_id


def cmd_down(args) -> int:
    state = read_state()
    pod_id = _running_pod_id(state)
    if not pod_id:
        return 0

    spend = _terminate(_client(), pod_id, state)
    print(f"terminated {pod_id} — spend: {_fmt_spend(spend)}")
    return 0


def cmd_status(args) -> int:
    state = read_state()
    pod_id = _running_pod_id(state)
    if not pod_id:
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
    print(f"spend: {_fmt_spend(spend)}")
    return 0


def cmd_teardown(args) -> int:
    """Stop all billing. Terminates every pod unconditionally; a volume is
    only ever deleted with --delete-volume, since that irreversibly destroys
    every downloaded model. The last line printed always says plainly
    whether anything is still costing money."""
    client = _client()

    pods = client.list_pods()
    for pod in pods:
        print(f"terminating {pod['id']} ({pod.get('name', '?')})")
        client.terminate_pod(pod["id"])
    clear_state()
    if not pods:
        print("no pods running")

    volumes = client.list_volumes()
    if not volumes:
        print("no volumes — nothing is billing")
        return 0

    total = 0.0
    for v in volumes:
        size = v.get("size", 0)
        cost = size * _VOLUME_GB_MONTHLY_USD
        total += cost
        print(f"volume {v.get('id')}  {size} GB  ~${cost:.2f}/month")

    if not args.delete_volume:
        print(
            f"\nstill billing: ~${total:.2f}/month across {len(volumes)} volume(s). "
            "`comfy teardown --delete-volume` removes them — that destroys every "
            "downloaded model and cannot be undone."
        )
        return 0

    for v in volumes:
        print(f"deleting {v.get('id')} — models are gone")
        client.delete_volume(str(v["id"]))
    print("nothing is billing")
    return 0


def _slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:limit] or "run"


def _open_or_reuse_tunnel(
    client: Client, pod_id: str, state: dict
) -> tuple[ComfyUI, Tunnel | None]:
    """Reuse an already-open tunnel recorded in state (from `comfy up`, or a
    previous `run`) rather than opening a second `ssh -L` on the same local
    port -- which fails outright, since the port is already bound. The
    returned Tunnel is None exactly when we reused one, so the caller knows
    whether it owns that process's lifecycle (open fresh -> we close it;
    reused -> whoever opened it manages it, `down` included)."""
    tunnel_port = state.get("tunnel_port")
    if tunnel_port:
        return ComfyUI(f"http://127.0.0.1:{tunnel_port}"), None
    host, port, user = client.ssh_target(pod_id)
    tun = Tunnel(host, port, user)
    tun.__enter__()
    return ComfyUI(tun.url), tun


def _finish_run(
    client: Client, pod_id: str, state: dict, tun: Tunnel | None, *, keep: bool, started_here: bool
) -> None:
    """Terminate the pod, or leave it running for `--keep` -- persisting a
    freshly-opened tunnel first so a later `comfy down` can still find and
    close it, same as `comfy up` does. Extracted from cmd_run's `finally`
    block so the keep-vs-terminate branching isn't nested three levels deep
    inside a try/finally/for.
    """
    if not keep:
        spend = _terminate(client, pod_id, state, tun)
        print(f"terminated {pod_id} — spend {_fmt_spend(spend)}")
        return
    if tun is not None:
        # We opened this tunnel ourselves -- persist it so a later
        # `comfy down` can find and close it, same as `comfy up` does.
        merge_state(tunnel_pid=tun.pid, tunnel_port=tun.local_port)
    if started_here:
        print(f"pod {pod_id} left running (--keep). `comfy down` when finished.")


def cmd_run(args) -> int:
    # Validate the run file -- and only the run file -- before anything that
    # could touch the pod. A typo'd mode must never cost money.
    run = load_run(Path(args.run_file))
    graph = workflow.load(WORKFLOW_DIR / f"{run.mode}.json")
    if run.overrides:
        graph = workflow.apply_overrides(graph, run.overrides)

    infra = load_infra(INFRA_PATH)
    client = _client()

    state = read_state()
    pod_id = state.get("pod_id")
    started_here = pod_id is None
    if pod_id is None:
        pod_id, choice, availability, _ssh = _bring_up(client, infra)
        print(f"pod: {pod_id} (GPU: {choice.id}, availability: {availability})")
        state = read_state()  # _bring_up just replaced it wholesale
    else:
        print(f"reusing pod: {pod_id}")

    out_dir = Path("out") / f"{date.today().isoformat()}-{_slug(run.prompt)}"
    client_id = secrets.token_hex(8)

    comfy, tun = _open_or_reuse_tunnel(client, pod_id, state)
    try:
        comfy.wait_ready()

        uploaded = comfy.upload_image(run.input) if run.input else None
        seeds = workflow.seeds_for(run)
        print(f"queuing {len(seeds)} job(s) -> {out_dir}")

        # One job at a time, download-then-advance: if job i fails, every
        # job before it is already fully downloaded to disk, and the error
        # names exactly which job failed and why.
        for i, seed in enumerate(seeds, start=1):
            try:
                g = workflow.apply_run(graph, run, seed, uploaded)
                prompt_id = comfy.queue(g, client_id)
                comfy.wait_for_history(prompt_id)
                saved = [comfy.download(e, out_dir) for e in comfy.outputs_of(prompt_id)]
            except (WorkflowError, ComfyUIError) as e:
                raise ComfyUIError(
                    f"job {i}/{len(seeds)} (seed {seed}) failed: {e}"
                ) from e
            names = ", ".join(p.name for p in saved) or "(no output files)"
            print(f"  {i}/{len(seeds)} done (seed {seed}): {names}")
    finally:
        _finish_run(client, pod_id, state, tun, keep=args.keep, started_here=started_here)

    print(f"saved to {out_dir}")
    return 0


handlers = {
    "provision": cmd_provision,
    "up": cmd_up,
    "run": cmd_run,
    "down": cmd_down,
    "status": cmd_status,
    "teardown": cmd_teardown,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    handler = handlers.get(args.command)
    if handler is None:
        print(f"not implemented: {args.command}", file=sys.stderr)
        return 1
    try:
        return handler(args)
    except (ConfigError, RunpodError, TunnelError, ComfyUIError, WorkflowError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
