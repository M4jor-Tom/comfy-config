"""comfy — drive ComfyUI on a rented Runpod GPU."""

import argparse
import os
import sys
from pathlib import Path

from .config import ConfigError, load_infra
from .provision import provision
from .runpod_api import Client, RunpodError

COMMANDS = ("provision", "up", "run", "down", "status")

handlers = {}

INFRA_PATH = Path("comfy.yaml")


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
    try:
        infra = load_infra(INFRA_PATH)
        volume_id = provision(_client(), infra)
    except (ConfigError, RunpodError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"\nvolume_id: {volume_id}\nAdd that to {INFRA_PATH} before `comfy up`.")
    return 0


handlers["provision"] = cmd_provision


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    handler = handlers.get(args.command)
    if handler is None:
        print(f"not implemented: {args.command}", file=sys.stderr)
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
