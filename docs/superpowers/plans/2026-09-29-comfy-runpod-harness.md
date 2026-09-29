# ComfyUI-on-Runpod Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local CLI that rents a Runpod GPU, runs a batch of ComfyUI generations across five modes, downloads results, and tears the machine down.

**Architecture:** Six small Python modules behind a `comfy` CLI. `runpod_api.py` drives Runpod REST v2 over stdlib `urllib`. `tunnel.py` shells out to `ssh -L` so port 8188 is never public. `comfyui.py` speaks ComfyUI's HTTP API through that tunnel. `workflow.py` holds the only real logic — patching committed API-format JSON graphs by `_meta.title` — and is the one module with heavy unit tests. Models live on a persistent 75 GB network volume, provisioned once.

**Tech Stack:** Python 3.12 (from the flake — the host has no `python3` at all), PyYAML as the sole third-party dependency, stdlib `urllib.request`/`json`/`subprocess` for everything else, system `ssh`, Nix flake for the dev shell and `nix run`.

**Spec:** `docs/superpowers/specs/2026-09-29-comfy-runpod-harness-design.md`

## Global Constraints

- **Datacenter: `EU-RO-1`.** The only one pairing standard network volumes with better-than-LOW GPU stock.
- **GPU order:** `NVIDIA RTX PRO 4500 Blackwell` (template `wgd3p4n4o6`, CUDA 13) then `NVIDIA GeForce RTX 4090` (template `cw3nka7d08`, CUDA 12.8). Template is tied to the GPU's CUDA line — never mix them.
- **Volume:** 75 GB, standard tier, `EU-RO-1`, mounted at `/workspace`. Increase-only.
- **Ports: `["22/tcp"]` only.** Port 8188 is never exposed. Ports cannot be added to a running pod.
- **Control plane: `https://api.runpod.io/v2` exclusively.** Never `rest.runpod.io/v1` (retires 2026-11-15) and never GraphQL. Do not add `runpodctl` or the `runpod` PyPI SDK — both ride the dying APIs.
- **No server-side pod TTL exists in v2.** Cost safety comes from `comfy run` terminating on completion by default.
- **Zero custom ComfyUI nodes.** Core ComfyUI covers all five modes. Never add a custom node pack.
- **Dependencies:** PyYAML only. Anything else needs justification against stdlib first.
- **ComfyUI paths on pod:** install `/workspace/runpod-slim/ComfyUI`, models `/workspace/runpod-slim/ComfyUI/models/<type>/`.
- **Model source:** `Comfy-Org/*` mirrors only, so no HuggingFace token. All paths under `split_files/`.
- **Commits:** Conventional Commits. Branch from `develop`, never fast-forward merge.

## Task Order Rationale

Live-hardware work sits in the **middle**, not the start. `comfy provision` is itself a CLI
command, so the client has to exist before it can run; and the five workflow JSONs can only
be exported from a running ComfyUI. Order is therefore: flake → config → API client →
provision (live) → up/tunnel (live) → export workflows (live, human) → workflow patching
(pure) → run (live) → polish.

Tasks 4, 5, 6 and 8 **spend real money**. Each names its cost. Get explicit approval before
running them.

---

### Task 1: Flake and package skeleton

Nothing can run before this — the host has no `python3`.

**Files:**
- Create: `flake.nix`
- Create: `pyproject.toml`
- Create: `src/comfy_runpod/__init__.py`
- Create: `src/comfy_runpod/cli.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a `nix develop` shell with `python3.12` + `pyyaml` + `pytest` + `openssh` + `jq` on PATH; `nix run .# -- --help` prints usage; console entry point `comfy = comfy_runpod.cli:main`.

- [ ] **Step 1: Write `flake.nix`**

```nix
{
  description = "comfy-runpod — ComfyUI generation harness on Runpod";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (s: f nixpkgs.legacyPackages.${s});
    in {
      packages = forAll (pkgs: {
        default = pkgs.python312Packages.buildPythonApplication {
          pname = "comfy-runpod";
          version = "0.1.0";
          src = ./.;
          pyproject = true;
          build-system = [ pkgs.python312Packages.setuptools ];
          dependencies = [ pkgs.python312Packages.pyyaml ];
          # ponytail: ssh is called via subprocess, so it must be on PATH at runtime
          makeWrapperArgs = [ "--prefix PATH : ${pkgs.openssh}/bin" ];
          doCheck = false;
        };
      });

      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            (python312.withPackages (ps: with ps; [ pyyaml pytest ]))
            openssh
            jq
          ];
          shellHook = ''
            export PYTHONPATH=$PWD/src:$PYTHONPATH
            echo "comfy-runpod dev shell — RUNPOD_API_KEY ${if true then "" else ""}\${RUNPOD_API_KEY:+is set}"
          '';
        };
      });
    };
}
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "comfy-runpod"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pyyaml"]

[project.scripts]
comfy = "comfy_runpod.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write the minimal CLI**

`src/comfy_runpod/__init__.py` is empty. `src/comfy_runpod/cli.py`:

```python
"""comfy — drive ComfyUI on a rented Runpod GPU."""

import argparse
import sys

COMMANDS = ("provision", "up", "run", "down", "status")


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    print(f"not implemented: {args.command}", file=sys.stderr)
    return 1
```

- [ ] **Step 4: Verify the shell and entry point both work**

Run:
```bash
nix develop -c python -c 'import yaml, sys; print(sys.version.split()[0], yaml.__version__)'
nix develop -c python -m comfy_runpod.cli --help
```
Expected: a Python 3.12.x version and a PyYAML version on the first; the usage block listing all five subcommands on the second. Both must exit 0.

- [ ] **Step 5: Verify `nix run` builds the package**

Run: `nix run .# -- --help`
Expected: same usage block. This proves `pyproject.toml` and the flake package agree. If it fails on `PYTHONPATH`, the packaging is wrong — fix it now, not later.

- [ ] **Step 6: Commit**

```bash
git add flake.nix flake.lock pyproject.toml src tests
git commit -m "feat(cli): flake dev shell and comfy command skeleton"
```

---

### Task 2: Config loading

**Files:**
- Create: `src/comfy_runpod/config.py`
- Create: `comfy.example.yaml`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `load_infra(path: Path) -> Infra` where `Infra` is a frozen dataclass with fields `datacenter: str`, `volume_id: str | None`, `gpus: list[GpuChoice]`, `terminate_after_seconds: int`.
  - `GpuChoice` frozen dataclass: `id: str`, `template: str`.
  - `load_run(path: Path) -> Run` where `Run` is a frozen dataclass: `mode: str`, `prompt: str`, `negative: str`, `count: int`, `size: tuple[int, int]`, `seed: int | None` (`None` means random), `input: Path | None`, `overrides: dict[str, object]`.
  - `ConfigError(Exception)`.
  - `MODES: tuple[str, ...] = ("t2i", "i2i", "t2v", "i2v", "v2v")`.
  - `MODES_NEEDING_INPUT: frozenset[str] = frozenset({"i2i", "i2v", "v2v"})`.
  - `state_path() -> Path`, `read_state() -> dict`, `write_state(d: dict) -> None`, `clear_state() -> None` — a JSON file at `.comfy-state.json` tracking the live pod.

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:

```python
import json
from pathlib import Path

import pytest

from comfy_runpod.config import (
    ConfigError,
    load_infra,
    load_run,
    read_state,
    write_state,
    clear_state,
)

INFRA = """
datacenter: EU-RO-1
volume_id: vol_abc123
gpus:
  - { id: "NVIDIA RTX PRO 4500 Blackwell", template: "wgd3p4n4o6" }
  - { id: "NVIDIA GeForce RTX 4090",       template: "cw3nka7d08" }
terminate_after: 3h
"""


def test_load_infra_parses_gpu_order(tmp_path):
    p = tmp_path / "comfy.yaml"
    p.write_text(INFRA)
    infra = load_infra(p)
    assert infra.datacenter == "EU-RO-1"
    assert infra.volume_id == "vol_abc123"
    assert [g.id for g in infra.gpus] == [
        "NVIDIA RTX PRO 4500 Blackwell",
        "NVIDIA GeForce RTX 4090",
    ]
    assert infra.gpus[0].template == "wgd3p4n4o6"
    assert infra.terminate_after_seconds == 10800


@pytest.mark.parametrize(
    "value,seconds", [("90s", 90), ("45m", 2700), ("2h", 7200), ("3600", 3600)]
)
def test_terminate_after_units(tmp_path, value, seconds):
    p = tmp_path / "comfy.yaml"
    p.write_text(INFRA.replace("terminate_after: 3h", f"terminate_after: {value}"))
    assert load_infra(p).terminate_after_seconds == seconds


def test_load_infra_rejects_empty_gpu_list(tmp_path):
    p = tmp_path / "comfy.yaml"
    p.write_text("datacenter: EU-RO-1\ngpus: []\n")
    with pytest.raises(ConfigError, match="at least one GPU"):
        load_infra(p)


def test_load_infra_rejects_gpu_missing_template(tmp_path):
    p = tmp_path / "comfy.yaml"
    p.write_text('datacenter: EU-RO-1\ngpus:\n  - { id: "X" }\n')
    with pytest.raises(ConfigError, match="template"):
        load_infra(p)


def test_volume_id_absent_is_none(tmp_path):
    """provision runs before a volume exists, so this must not raise."""
    p = tmp_path / "comfy.yaml"
    p.write_text('datacenter: EU-RO-1\ngpus:\n  - { id: "X", template: "t" }\n')
    assert load_infra(p).volume_id is None


RUN = """
mode: t2i
prompt: "a fox in a misty forest"
negative: "blurry"
count: 16
size: [1024, 1024]
seed: random
overrides:
  sampler.cfg: 3.5
"""


def test_load_run_parses(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(RUN)
    run = load_run(p)
    assert run.mode == "t2i"
    assert run.count == 16
    assert run.size == (1024, 1024)
    assert run.seed is None  # "random" normalises to None
    assert run.overrides == {"sampler.cfg": 3.5}
    assert run.input is None


def test_load_run_fixed_seed(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(RUN.replace("seed: random", "seed: 42"))
    assert load_run(p).seed == 42


def test_load_run_rejects_unknown_mode(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(RUN.replace("mode: t2i", "mode: t2x"))
    with pytest.raises(ConfigError, match="t2x"):
        load_run(p)


def test_load_run_requires_input_for_i2i(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(RUN.replace("mode: t2i", "mode: i2i"))
    with pytest.raises(ConfigError, match="input"):
        load_run(p)


def test_load_run_resolves_input_relative_to_run_file(tmp_path):
    img = tmp_path / "cat.png"
    img.write_bytes(b"x")
    p = tmp_path / "run.yaml"
    p.write_text(RUN.replace("mode: t2i", "mode: i2i") + "input: cat.png\n")
    assert load_run(p).input == img


def test_load_run_rejects_missing_input_file(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(RUN.replace("mode: t2i", "mode: i2i") + "input: nope.png\n")
    with pytest.raises(ConfigError, match="nope.png"):
        load_run(p)


def test_load_run_rejects_zero_count(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(RUN.replace("count: 16", "count: 0"))
    with pytest.raises(ConfigError, match="count"):
        load_run(p)


def test_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert read_state() == {}
    write_state({"pod_id": "abc", "started": 123})
    assert read_state()["pod_id"] == "abc"
    clear_state()
    assert read_state() == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nix develop -c python -m pytest tests/test_config.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'comfy_runpod.config'`.

- [ ] **Step 3: Write `src/comfy_runpod/config.py`**

```python
"""Load and validate the two config files, plus local pod state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MODES: tuple[str, ...] = ("t2i", "i2i", "t2v", "i2v", "v2v")
MODES_NEEDING_INPUT: frozenset[str] = frozenset({"i2i", "i2v", "v2v"})

_UNITS = {"s": 1, "m": 60, "h": 3600}


class ConfigError(Exception):
    """A config file is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class GpuChoice:
    id: str
    template: str


@dataclass(frozen=True)
class Infra:
    datacenter: str
    gpus: list[GpuChoice]
    volume_id: str | None = None
    terminate_after_seconds: int = 10800


@dataclass(frozen=True)
class Run:
    mode: str
    prompt: str
    count: int
    size: tuple[int, int]
    negative: str = ""
    seed: int | None = None
    input: Path | None = None
    overrides: dict[str, object] = field(default_factory=dict)


def _load_yaml(path: Path) -> dict:
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as e:
        raise ConfigError(f"{path} does not exist") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return raw


def _duration_seconds(value: object) -> int:
    """Accept 3h / 45m / 90s / bare seconds."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    mult = _UNITS.get(text[-1:], None)
    if mult is None:
        try:
            return int(text)
        except ValueError as e:
            raise ConfigError(f"terminate_after {value!r} is not a duration") from e
    try:
        return int(text[:-1]) * mult
    except ValueError as e:
        raise ConfigError(f"terminate_after {value!r} is not a duration") from e


def load_infra(path: Path) -> Infra:
    raw = _load_yaml(path)
    if "datacenter" not in raw:
        raise ConfigError(f"{path}: datacenter is required")

    gpus_raw = raw.get("gpus") or []
    if not gpus_raw:
        raise ConfigError(f"{path}: at least one GPU must be listed under 'gpus'")
    gpus = []
    for entry in gpus_raw:
        if not isinstance(entry, dict) or "id" not in entry:
            raise ConfigError(f"{path}: each gpus entry needs an 'id'")
        if "template" not in entry:
            raise ConfigError(
                f"{path}: gpu {entry['id']!r} needs a 'template' — the ComfyUI "
                "template must match the GPU's CUDA line"
            )
        gpus.append(GpuChoice(id=str(entry["id"]), template=str(entry["template"])))

    return Infra(
        datacenter=str(raw["datacenter"]),
        gpus=gpus,
        volume_id=(str(raw["volume_id"]) if raw.get("volume_id") else None),
        terminate_after_seconds=_duration_seconds(raw.get("terminate_after", "3h")),
    )


def load_run(path: Path) -> Run:
    raw = _load_yaml(path)

    mode = str(raw.get("mode", "")).strip()
    if mode not in MODES:
        raise ConfigError(f"{path}: mode {mode!r} is not one of {', '.join(MODES)}")

    prompt = str(raw.get("prompt", "")).strip()
    if not prompt:
        raise ConfigError(f"{path}: prompt is required and must not be empty")

    count = raw.get("count", 1)
    if not isinstance(count, int) or count < 1:
        raise ConfigError(f"{path}: count must be a positive integer, got {count!r}")

    size = raw.get("size", [1024, 1024])
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        raise ConfigError(f"{path}: size must be [width, height], got {size!r}")
    width, height = int(size[0]), int(size[1])

    seed_raw = raw.get("seed", "random")
    seed = None if str(seed_raw) == "random" else int(seed_raw)

    input_path: Path | None = None
    if raw.get("input"):
        input_path = (path.parent / str(raw["input"])).resolve()
        if not input_path.is_file():
            raise ConfigError(f"{path}: input file {raw['input']!r} not found")
    elif mode in MODES_NEEDING_INPUT:
        raise ConfigError(f"{path}: mode {mode!r} requires an 'input' file")

    overrides = raw.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ConfigError(f"{path}: overrides must be a mapping")

    return Run(
        mode=mode,
        prompt=prompt,
        negative=str(raw.get("negative", "")),
        count=count,
        size=(width, height),
        seed=seed,
        input=input_path,
        overrides=overrides,
    )


def state_path() -> Path:
    return Path(".comfy-state.json")


def read_state() -> dict:
    p = state_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def write_state(d: dict) -> None:
    state_path().write_text(json.dumps(d, indent=2))


def clear_state() -> None:
    state_path().unlink(missing_ok=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `nix develop -c python -m pytest tests/test_config.py -q`
Expected: all pass (15 tests).

- [ ] **Step 5: Write `comfy.example.yaml`**

```yaml
# Copy to comfy.yaml and fill volume_id after `comfy provision`.
datacenter: EU-RO-1

# Set by `comfy provision`. Leave blank the first time.
volume_id:

# Ordered: the first GPU with capacity wins. The template MUST match the
# GPU's CUDA line — Blackwell needs CUDA 13, Ada needs CUDA 12.8.
gpus:
  - { id: "NVIDIA RTX PRO 4500 Blackwell", template: "wgd3p4n4o6" }  # 32 GB, $0.72/hr
  - { id: "NVIDIA GeForce RTX 4090",       template: "cw3nka7d08" }  # 24 GB, $0.74/hr

# Local watchdog only. Runpod REST v2 has no server-side pod TTL, so this
# dies if this machine does. The real guard is `comfy run` terminating on
# completion.
terminate_after: 3h
```

- [ ] **Step 6: Add state file to gitignore and commit**

```bash
echo '.comfy-state.json' >> .gitignore
echo 'comfy.yaml' >> .gitignore
git add src/comfy_runpod/config.py tests/test_config.py comfy.example.yaml .gitignore
git commit -m "feat(config): load and validate infra and run files"
```

`comfy.yaml` is gitignored because it will carry a real volume ID; the example is committed.

---

### Task 3: Runpod REST v2 client

No cloud calls in the tests — every test drives a fake transport. This task must be
correct before Task 4 spends money.

**Files:**
- Create: `src/comfy_runpod/runpod_api.py`
- Create: `tests/test_runpod_api.py`

**Interfaces:**
- Consumes: `config.GpuChoice`.
- Produces:
  - `RunpodError(Exception)`.
  - `Client(api_key: str, transport: Callable[[str, str, dict | None], dict] | None = None)`.
  - `Client.pick_gpu(gpus: list[GpuChoice], datacenter: str) -> tuple[GpuChoice, str]` — returns the first choice with availability other than `NONE` in that datacenter, plus the availability level. Raises `RunpodError` if none.
  - `Client.create_volume(name: str, size_gb: int, datacenter: str) -> str` — returns volume id.
  - `Client.create_pod(*, name, template_id, datacenter, volume_id, gpu_id=None, cpu=None) -> dict` — exactly one of `gpu_id`/`cpu`. `volume_id` accepts `None`, which omits `mounts` entirely so a volume-less pod can be created (see the volume-optional note below).
  - `Client.get_pod(pod_id: str) -> dict`.
  - `Client.terminate_pod(pod_id: str) -> None`.
  - `Client.pod_spend(pod_id: str) -> float | None` — real billed USD, `None` if unavailable.
  - `Client.ssh_target(pod_id: str) -> tuple[str, int, str]` — `(host, port, user)` for direct-TCP SSH. Raises `RunpodError` while the pod has no runtime yet.

- [ ] **Step 1: Write the failing tests**

`tests/test_runpod_api.py`:

```python
import pytest

from comfy_runpod.config import GpuChoice
from comfy_runpod.runpod_api import Client, RunpodError

BLACKWELL = GpuChoice(id="NVIDIA RTX PRO 4500 Blackwell", template="wgd3p4n4o6")
ADA = GpuChoice(id="NVIDIA GeForce RTX 4090", template="cw3nka7d08")

CATALOG = {
    "gpus": [
        {
            "id": "NVIDIA RTX PRO 4500 Blackwell",
            "memory": 32,
            "dataCenters": [{"id": "EU-RO-1", "availability": "MEDIUM"}],
        },
        {
            "id": "NVIDIA GeForce RTX 4090",
            "memory": 24,
            "dataCenters": [
                {"id": "EU-RO-1", "availability": "HIGH"},
                {"id": "US-IL-1", "availability": "LOW"},
            ],
        },
    ]
}


class FakeTransport:
    """Records calls and replays canned responses."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        for key, value in self.responses.items():
            if key in path:
                return value
        return {}


def test_pick_gpu_prefers_first_with_capacity():
    t = FakeTransport({"/catalog/gpus": CATALOG})
    chosen, avail = Client("k", transport=t).pick_gpu([BLACKWELL, ADA], "EU-RO-1")
    assert chosen is BLACKWELL
    assert avail == "MEDIUM"


def test_pick_gpu_falls_through_when_primary_absent_from_datacenter():
    catalog = {
        "gpus": [
            {"id": BLACKWELL.id, "dataCenters": [{"id": "US-IL-1", "availability": "HIGH"}]},
            {"id": ADA.id, "dataCenters": [{"id": "EU-RO-1", "availability": "HIGH"}]},
        ]
    }
    t = FakeTransport({"/catalog/gpus": catalog})
    chosen, _ = Client("k", transport=t).pick_gpu([BLACKWELL, ADA], "EU-RO-1")
    assert chosen is ADA


def test_pick_gpu_skips_none_availability():
    catalog = {
        "gpus": [
            {"id": BLACKWELL.id, "dataCenters": [{"id": "EU-RO-1", "availability": "NONE"}]},
            {"id": ADA.id, "dataCenters": [{"id": "EU-RO-1", "availability": "LOW"}]},
        ]
    }
    t = FakeTransport({"/catalog/gpus": catalog})
    chosen, avail = Client("k", transport=t).pick_gpu([BLACKWELL, ADA], "EU-RO-1")
    assert chosen is ADA
    assert avail == "LOW"


def test_pick_gpu_raises_when_nothing_available():
    catalog = {"gpus": [{"id": BLACKWELL.id, "dataCenters": []}]}
    t = FakeTransport({"/catalog/gpus": catalog})
    with pytest.raises(RunpodError, match="no configured GPU"):
        Client("k", transport=t).pick_gpu([BLACKWELL], "EU-RO-1")


def test_create_pod_body_shape_for_gpu():
    t = FakeTransport({"/pods": {"id": "pod1"}})
    Client("k", transport=t).create_pod(
        name="comfy",
        template_id="wgd3p4n4o6",
        datacenter="EU-RO-1",
        volume_id="vol_abc",
        gpu_id=BLACKWELL.id,
    )
    method, path, body = t.calls[-1]
    assert (method, path) == ("POST", "/pods")
    assert body["gpu"] == {"id": BLACKWELL.id, "count": 1}
    assert body["templateId"] == "wgd3p4n4o6"
    assert body["cloud"] == "SECURE"
    assert body["dataCenterIds"] == ["EU-RO-1"]
    assert body["ports"] == ["22/tcp"]          # 8188 must never be exposed
    assert body["startSsh"] is True
    assert body["mounts"] == {"network": [{"volumeId": "vol_abc", "path": "/workspace"}]}
    assert "disk" not in body                   # let the template's value stand
    assert "cpu" not in body
    assert "terminateAfter" not in body         # no such field in v2


def test_create_pod_body_shape_for_cpu():
    t = FakeTransport({"/pods": {"id": "pod1"}})
    Client("k", transport=t).create_pod(
        name="provision",
        template_id=None,
        datacenter="EU-RO-1",
        volume_id="vol_abc",
        cpu={"id": "cpu5c", "vcpuCount": 2},
    )
    _, _, body = t.calls[-1]
    assert body["cpu"] == {"id": "cpu5c", "vcpuCount": 2}
    assert body["image"]                        # CPU pod needs an image, not a template
    assert "gpu" not in body
    assert "templateId" not in body


def test_create_pod_rejects_both_gpu_and_cpu():
    t = FakeTransport({})
    with pytest.raises(RunpodError, match="exactly one"):
        Client("k", transport=t).create_pod(
            name="x", template_id=None, datacenter="EU-RO-1", volume_id="v",
            gpu_id="g", cpu={"id": "cpu5c", "vcpuCount": 2},
        )


def test_create_pod_rejects_neither_gpu_nor_cpu():
    t = FakeTransport({})
    with pytest.raises(RunpodError, match="exactly one"):
        Client("k", transport=t).create_pod(
            name="x", template_id=None, datacenter="EU-RO-1", volume_id="v",
        )


def test_create_pod_omits_mounts_when_no_volume():
    """The harness must still work for a user who keeps no standing volume."""
    t = FakeTransport({"/pods": {"id": "pod1"}})
    Client("k", transport=t).create_pod(
        name="comfy", template_id="wgd3p4n4o6", datacenter="EU-RO-1",
        volume_id=None, gpu_id=BLACKWELL.id,
    )
    _, _, body = t.calls[-1]
    assert "mounts" not in body
    assert body["ports"] == ["22/tcp"]


def test_create_volume_body_shape():
    t = FakeTransport({"/network-volumes": {"id": "vol_new"}})
    vid = Client("k", transport=t).create_volume("comfy-models", 75, "EU-RO-1")
    assert vid == "vol_new"
    method, path, body = t.calls[-1]
    assert (method, path) == ("POST", "/network-volumes")
    assert body == {"name": "comfy-models", "size": 75, "dataCenter": "EU-RO-1"}


def test_terminate_uses_action_endpoint():
    t = FakeTransport({})
    Client("k", transport=t).terminate_pod("pod1")
    assert t.calls[-1] == ("POST", "/pods/pod1/action", {"action": "terminate"})


def test_ssh_target_reads_direct_tcp_mapping():
    pod = {
        "id": "pod1",
        "desiredStatus": "RUNNING",
        "runtime": {"ports": [{"private": 22, "public": 40022, "ip": "1.2.3.4", "type": "tcp"}]},
    }
    t = FakeTransport({"/pods/pod1": pod})
    host, port, user = Client("k", transport=t).ssh_target("pod1")
    assert (host, port, user) == ("1.2.3.4", 40022, "root")


def test_ssh_target_raises_while_runtime_is_null():
    t = FakeTransport({"/pods/pod1": {"id": "pod1", "runtime": None}})
    with pytest.raises(RunpodError, match="not ready"):
        Client("k", transport=t).ssh_target("pod1")


def test_ssh_target_raises_when_no_tcp_22_mapping():
    pod = {"runtime": {"ports": [{"private": 8188, "public": 1, "ip": "1.2.3.4", "type": "http"}]}}
    t = FakeTransport({"/pods/pod1": pod})
    with pytest.raises(RunpodError, match="22/tcp"):
        Client("k", transport=t).ssh_target("pod1")


def test_pod_spend_sums_billing_records():
    billing = {"records": [{"amount": 0.21}, {"amount": 0.09}]}
    t = FakeTransport({"/billing/pods": billing})
    assert Client("k", transport=t).pod_spend("pod1") == pytest.approx(0.30)


def test_pod_spend_returns_none_when_unavailable():
    t = FakeTransport({"/billing/pods": {}})
    assert Client("k", transport=t).pod_spend("pod1") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nix develop -c python -m pytest tests/test_runpod_api.py -q`
Expected: `ModuleNotFoundError: No module named 'comfy_runpod.runpod_api'`.

- [ ] **Step 3: Write `src/comfy_runpod/runpod_api.py`**

```python
"""Runpod REST v2 client.

v2 only. REST v1 retires 2026-11-15 and GraphQL in early 2027, which is why
this does not use runpodctl or the runpod PyPI SDK — both still ride those.
Schema: https://api.runpod.io/v2/openapi.json (public, no auth needed).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable

from .config import GpuChoice

API_BASE = "https://api.runpod.io/v2"
VOLUME_MOUNT_PATH = "/workspace"

# Minimal Ubuntu image for the provisioning CPU pod. It only needs curl and sshd,
# both of which Runpod's base image provides.
CPU_POD_IMAGE = "runpod/base:1.0.2-ubuntu2404"


class RunpodError(Exception):
    """A Runpod API call failed, or returned something unusable."""


def _http(method: str, path: str, body: dict | None, api_key: str) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RunpodError(f"{method} {path} failed with {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RunpodError(f"{method} {path} could not reach Runpod: {e.reason}") from e
    return json.loads(raw) if raw else {}


class Client:
    def __init__(
        self,
        api_key: str,
        transport: Callable[[str, str, dict | None], dict] | None = None,
    ) -> None:
        if not api_key:
            raise RunpodError("RUNPOD_API_KEY is not set")
        self._key = api_key
        self._transport = transport or (
            lambda m, p, b=None: _http(m, p, b, api_key)
        )

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        return self._transport(method, path, body)

    # --- catalog ---------------------------------------------------------

    def pick_gpu(
        self, gpus: list[GpuChoice], datacenter: str
    ) -> tuple[GpuChoice, str]:
        catalog = self._call(
            "GET", "/catalog/gpus?include=AVAILABILITY&product=POD"
        )
        by_id = {g.get("id"): g for g in catalog.get("gpus") or []}
        for choice in gpus:
            entry = by_id.get(choice.id)
            if not entry:
                continue
            for dc in entry.get("dataCenters") or []:
                if dc.get("id") == datacenter and dc.get("availability") != "NONE":
                    return choice, str(dc.get("availability"))
        wanted = ", ".join(g.id for g in gpus)
        raise RunpodError(
            f"no configured GPU has capacity in {datacenter}. Wanted: {wanted}. "
            "Check the catalog, or add another GPU to comfy.yaml."
        )

    # --- volumes ---------------------------------------------------------

    def create_volume(self, name: str, size_gb: int, datacenter: str) -> str:
        out = self._call(
            "POST",
            "/network-volumes",
            {"name": name, "size": size_gb, "dataCenter": datacenter},
        )
        vid = out.get("id")
        if not vid:
            raise RunpodError(f"volume create returned no id: {out}")
        return str(vid)

    # --- pods ------------------------------------------------------------

    def create_pod(
        self,
        *,
        name: str,
        template_id: str | None,
        datacenter: str,
        volume_id: str,
        gpu_id: str | None = None,
        cpu: dict | None = None,
    ) -> dict:
        if bool(gpu_id) == bool(cpu):
            raise RunpodError("create_pod needs exactly one of gpu_id or cpu")

        body: dict = {
            "name": name,
            "cloud": "SECURE",  # network volumes are Secure Cloud only
            "dataCenterIds": [datacenter],
            "ports": ["22/tcp"],  # 8188 is reached over the tunnel, never exposed
            "startSsh": True,
        }
        if volume_id:
            body["mounts"] = {
                "network": [{"volumeId": volume_id, "path": VOLUME_MOUNT_PATH}]
            }
        if gpu_id:
            body["gpu"] = {"id": gpu_id, "count": 1}
            body["templateId"] = template_id
        else:
            body["cpu"] = cpu
            body["image"] = CPU_POD_IMAGE

        out = self._call("POST", "/pods", body)
        if not out.get("id"):
            raise RunpodError(f"pod create returned no id: {out}")
        return out

    def get_pod(self, pod_id: str) -> dict:
        return self._call("GET", f"/pods/{pod_id}")

    def terminate_pod(self, pod_id: str) -> None:
        self._call("POST", f"/pods/{pod_id}/action", {"action": "terminate"})

    def ssh_target(self, pod_id: str) -> tuple[str, int, str]:
        """Direct-TCP SSH endpoint. Runpod's proxy SSH cannot forward ports."""
        pod = self.get_pod(pod_id)
        runtime = pod.get("runtime")
        if not runtime:
            raise RunpodError(f"pod {pod_id} is not ready — runtime is still null")
        for p in runtime.get("ports") or []:
            if p.get("private") == 22 and p.get("public") and p.get("ip"):
                return str(p["ip"]), int(p["public"]), "root"
        raise RunpodError(
            f"pod {pod_id} has no 22/tcp mapping. Ports are fixed at creation — "
            "the pod must be recreated with 22/tcp exposed."
        )

    def pod_spend(self, pod_id: str) -> float | None:
        try:
            out = self._call("GET", f"/billing/pods?podId={pod_id}")
        except RunpodError:
            return None
        records = out.get("records")
        if not records:
            return None
        return sum(float(r.get("amount", 0.0)) for r in records)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `nix develop -c python -m pytest tests/test_runpod_api.py -q`
Expected: all pass (15 tests).

- [ ] **Step 5: Commit**

```bash
git add src/comfy_runpod/runpod_api.py tests/test_runpod_api.py
git commit -m "feat(runpod): REST v2 client for pods, volumes and billing"
```

---

### Task 4: `comfy provision` — live, costs about $0.15

**⚠️ Spends real money and creates a standing $5.25/month charge. Get explicit approval
before Step 4.** The volume persists until deliberately deleted.

**Files:**
- Create: `src/comfy_runpod/provision.py`
- Create: `scripts/download-models.sh`
- Modify: `src/comfy_runpod/cli.py`
- Create: `tests/test_provision.py`

**Interfaces:**
- Consumes: `runpod_api.Client`, `config.Infra`.
- Produces:
  - `MODEL_FILES: tuple[ModelFile, ...]` where `ModelFile` is a frozen dataclass `repo: str`, `path: str`, `target_dir: str`, `gb: float`.
  - `total_gb() -> float`.
  - `render_download_script() -> str` — a self-contained bash script.
  - `provision(client, infra, volume_size_gb=75) -> str` — returns the volume id.

- [ ] **Step 1: Write the failing tests**

`tests/test_provision.py`:

```python
import pytest

from comfy_runpod.provision import MODEL_FILES, render_download_script, total_gb


def test_model_set_covers_all_five_modes_within_budget():
    assert total_gb() == pytest.approx(48.7, abs=0.2)


def test_every_model_comes_from_a_comfy_org_mirror():
    """Comfy-Org mirrors need no HuggingFace token. Anything else would."""
    for m in MODEL_FILES:
        assert m.repo.startswith("Comfy-Org/"), m.repo


def test_quality_ruling_uses_non_distilled_bf16_z_image():
    paths = [m.path for m in MODEL_FILES]
    assert any("z_image_bf16" in p for p in paths)
    assert not any("turbo" in p for p in paths)
    assert not any("int8" in p for p in paths)


def test_target_dirs_are_known_comfyui_model_types():
    allowed = {"diffusion_models", "text_encoders", "vae", "loras"}
    assert {m.target_dir for m in MODEL_FILES} <= allowed


def test_no_duplicate_targets():
    seen = [(m.target_dir, m.path.rsplit("/", 1)[-1]) for m in MODEL_FILES]
    assert len(seen) == len(set(seen))


def test_script_is_strict_bash_and_resumes():
    s = render_download_script()
    assert s.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in s
    assert "--continue" in s or "-C -" in s  # resumable download
    assert "/workspace/runpod-slim/ComfyUI/models" in s


def test_script_mentions_every_model_file():
    s = render_download_script()
    for m in MODEL_FILES:
        assert m.path in s


def test_script_verifies_sizes_rather_than_trusting_exit_code():
    s = render_download_script()
    assert "stat" in s or "du " in s
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nix develop -c python -m pytest tests/test_provision.py -q`
Expected: `ModuleNotFoundError: No module named 'comfy_runpod.provision'`.

- [ ] **Step 3: Write `src/comfy_runpod/provision.py`**

```python
"""One-time: create the volume and fill it with models via a cheap CPU pod.

A CPU pod costs $0.035/vCPU/hr against $0.72/hr for the GPU, and network
volumes attach to CPU pods. So a slow download host costs pennies instead of
dollars — even a 9-hour worst case is about $0.63, once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Infra
from .runpod_api import Client, RunpodError

MODELS_ROOT = "/workspace/runpod-slim/ComfyUI/models"

Z_IMAGE = "Comfy-Org/z_image"
WAN22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"


@dataclass(frozen=True)
class ModelFile:
    repo: str
    path: str
    target_dir: str
    gb: float


# Verified against the HuggingFace API on 2026-09-29. Sizes are decimal GB.
# Non-distilled bf16 Z-Image per the quality-at-equal-cost ruling (spec 13.2).
MODEL_FILES: tuple[ModelFile, ...] = (
    # t2i and i2i share these three.
    ModelFile(Z_IMAGE, "split_files/diffusion_models/z_image_bf16.safetensors",
              "diffusion_models", 12.3),
    ModelFile(Z_IMAGE, "split_files/text_encoders/qwen_3_4b.safetensors",
              "text_encoders", 8.0),
    ModelFile(Z_IMAGE, "split_files/vae/ae.safetensors", "vae", 0.3),
    # t2v and i2v: one model does both.
    ModelFile(WAN22, "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
              "diffusion_models", 10.0),
    # v2v.
    ModelFile(WAN22, "split_files/diffusion_models/wan2.2_fun_control_5B_bf16.safetensors",
              "diffusion_models", 10.0),
    # Shared across every Wan mode.
    ModelFile(WAN22, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
              "text_encoders", 6.7),
    ModelFile(WAN22, "split_files/vae/wan2.2_vae.safetensors", "vae", 1.4),
)


def total_gb() -> float:
    return round(sum(m.gb for m in MODEL_FILES), 1)


def render_download_script() -> str:
    """Bash that runs on the pod. curl only — no Python, no hf CLI needed."""
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"ROOT={MODELS_ROOT}",
        'mkdir -p "$ROOT"/{diffusion_models,text_encoders,vae,loras}',
        "",
        "started=$(date +%s)",
        "",
    ]
    for m in MODEL_FILES:
        name = m.path.rsplit("/", 1)[-1]
        url = f"https://huggingface.co/{m.repo}/resolve/main/{m.path}"
        dest = f'"$ROOT/{m.target_dir}/{name}"'
        lines += [
            f"echo '==> {name} ({m.gb} GB)'",
            # --continue-at - resumes a partial file, so a retry is cheap.
            f"curl -fL --retry 5 --retry-delay 5 --continue-at - "
            f"-o {dest} '{url}'",
            # Trust measured bytes, not curl's exit code.
            f'actual=$(stat -c %s {dest})',
            f'min=$(python3 -c "print(int({m.gb} * 1e9 * 0.95))" 2>/dev/null '
            f'|| echo {int(m.gb * 0.95 * 1e9)})',
            'if [ "$actual" -lt "$min" ]; then',
            f'  echo "FAIL: {name} is $actual bytes, expected ~{m.gb} GB" >&2; exit 1',
            "fi",
            "",
        ]
    lines += [
        "elapsed=$(( $(date +%s) - started ))",
        f'echo "=== done: {total_gb()} GB in ${{elapsed}}s "'
        f'"($(python3 -c "print(round({total_gb()}*1000/$elapsed,1))" '
        f'2>/dev/null || echo ?) MB/s) ==="',
        f'du -sh {MODELS_ROOT}',
    ]
    return "\n".join(lines) + "\n"


def provision(client: Client, infra: Infra, volume_size_gb: int = 75) -> str:
    """Create the volume if absent, then fill it from a CPU pod. Returns volume id."""
    volume_id = infra.volume_id
    if volume_id:
        print(f"reusing existing volume {volume_id}")
    else:
        volume_id = client.create_volume(
            "comfy-models", volume_size_gb, infra.datacenter
        )
        print(f"created volume {volume_id} ({volume_size_gb} GB) — PUT THIS IN comfy.yaml")

    pod = client.create_pod(
        name="comfy-provision",
        template_id=None,
        datacenter=infra.datacenter,
        volume_id=volume_id,
        cpu={"id": "cpu5c", "vcpuCount": 2},
    )
    pod_id = str(pod["id"])
    print(f"provisioning pod {pod_id} — terminating it is this function's job")
    try:
        _wait_for_ssh(client, pod_id)
        host, port, user = client.ssh_target(pod_id)
        print(f"downloading {total_gb()} GB — ssh {user}@{host} -p {port}")
        _run_download(host, port, user)
    finally:
        client.terminate_pod(pod_id)
        print(f"terminated {pod_id}")
    return volume_id


def _wait_for_ssh(client: Client, pod_id: str, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.ssh_target(pod_id)
            return
        except RunpodError:
            time.sleep(10)
    raise RunpodError(f"pod {pod_id} never exposed SSH within {timeout}s")


def _run_download(host: str, port: int, user: str) -> None:
    import subprocess

    script = render_download_script()
    cmd = [
        "ssh", "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        f"{user}@{host}", "bash -s",
    ]
    result = subprocess.run(cmd, input=script, text=True)
    if result.returncode != 0:
        raise RunpodError(f"model download failed with exit {result.returncode}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `nix develop -c python -m pytest tests/test_provision.py -q`
Expected: all pass (8 tests).

- [ ] **Step 5: Wire `provision` into the CLI**

In `src/comfy_runpod/cli.py`, replace the body of `main` with a dispatch table and add
the provision handler:

```python
import os
from pathlib import Path

from .config import ConfigError, load_infra
from .runpod_api import Client, RunpodError

INFRA_PATH = Path("comfy.yaml")


def _client() -> Client:
    return Client(os.environ.get("RUNPOD_API_KEY", ""))


def cmd_provision(args) -> int:
    infra = load_infra(INFRA_PATH)
    volume_id = provision(_client(), infra)
    print(f"\nvolume_id: {volume_id}\nAdd that to {INFRA_PATH} before `comfy up`.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    handlers = {"provision": cmd_provision}
    handler = handlers.get(args.command)
    if handler is None:
        print(f"not implemented: {args.command}", file=sys.stderr)
        return 1
    try:
        return handler(args)
    except (ConfigError, RunpodError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
```

Add `from .provision import provision` to the imports.

- [ ] **Step 6: Dry-run the script without spending anything**

Run: `nix develop -c python -c 'from comfy_runpod.provision import render_download_script as r; print(r())' | bash -n`
Expected: no output, exit 0 — `bash -n` proves the generated script parses before we
ship it to a paid machine.

- [ ] **Step 7: 🔴 LIVE — run provisioning (~$0.15, plus $5.25/month thereafter)**

Requires approval. Requires `comfy.yaml` copied from the example with `volume_id` blank.

```bash
nix develop -c comfy provision
```

Expected: a volume id printed, a CPU pod created, seven files downloaded with a final
`=== done: 48.7 GB in Ns (X MB/s) ===` line, then the pod terminated. **Record the
observed MB/s in the plan's completion notes** — it is the first real datapoint for this
account and it settles spec §3's central assumption.

Then put the volume id in `comfy.yaml`.

- [ ] **Step 8: Confirm no pod is left running**

Run: `nix develop -c python -c "
import os; from comfy_runpod.runpod_api import Client
print([(p['id'], p.get('desiredStatus')) for p in Client(os.environ['RUNPOD_API_KEY'])._call('GET','/pods').get('pods',[])])"`
Expected: `[]`, or no pod named `comfy-provision`. If one survives, terminate it now —
a leaked CPU pod is cheap but a leaked habit is not.

- [ ] **Step 9: Commit**

```bash
git add src/comfy_runpod/provision.py src/comfy_runpod/cli.py tests/test_provision.py
git commit -m "feat(provision): create volume and download 48.7 GB model set via CPU pod"
```

---

### Task 5: `comfy up` / `down` / `status` and the SSH tunnel — live, about $0.10

**⚠️ Boots a GPU pod. Step 6 is the first-ever live test of the CUDA 13 template.**

**Files:**
- Create: `src/comfy_runpod/tunnel.py`
- Create: `src/comfy_runpod/comfyui.py`
- Modify: `src/comfy_runpod/cli.py`
- Create: `tests/test_tunnel.py`

**Interfaces:**
- Consumes: `runpod_api.Client`, `config.Infra`, `config.read_state`/`write_state`/`clear_state`.
- Produces:
  - `tunnel.Tunnel(host, port, user, local_port=8188)` — context manager; `.url` is `http://127.0.0.1:<local_port>`; `ssh_command()` returns the exact argv list.
  - `tunnel.TunnelError(Exception)`.
  - `comfyui.ComfyUI(base_url: str)` with `.system_stats() -> dict`, `.wait_ready(timeout: int) -> dict`.
  - `comfyui.ComfyUIError(Exception)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tunnel.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nix develop -c python -m pytest tests/test_tunnel.py -q`
Expected: `ModuleNotFoundError: No module named 'comfy_runpod.tunnel'`.

- [ ] **Step 3: Write `src/comfy_runpod/tunnel.py`**

```python
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
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
                err = (self._proc.stderr.read() if self._proc.stderr else "")[:400]
                raise TunnelError(f"ssh exited immediately: {err}")
            with socket.socket() as s:
                s.settimeout(2)
                if s.connect_ex(("127.0.0.1", self.local_port)) == 0:
                    return
            time.sleep(1)
        self.close()
        raise TunnelError(
            f"port {self.local_port} never opened. Is something already using it?"
        )
```

- [ ] **Step 4: Write `src/comfy_runpod/comfyui.py` (readiness only for now)**

```python
"""ComfyUI HTTP API, reached through the SSH tunnel at 127.0.0.1."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class ComfyUIError(Exception):
    """ComfyUI was unreachable or returned something unusable."""


class ComfyUI:
    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")

    def _get_json(self, path: str, timeout: int = 30) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base}{path}", timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise ComfyUIError(f"GET {path} failed: {e}") from e

    def system_stats(self) -> dict:
        return self._get_json("/system_stats")

    def wait_ready(self, timeout: int = 900) -> dict:
        """Poll until ComfyUI serves. 'Running' is not ready — this is."""
        deadline = time.time() + timeout
        last = "no attempt yet"
        while time.time() < deadline:
            try:
                return self.system_stats()
            except ComfyUIError as e:
                last = str(e)
                time.sleep(10)
        raise ComfyUIError(f"ComfyUI not ready within {timeout}s. Last error: {last}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `nix develop -c python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: 🔴 LIVE — boot the GPU pod (~$0.10 for 8 minutes)**

Requires approval. Wire `up`, `down` and `status` into `cli.py` first (dispatch table
entries calling `Client.pick_gpu` → `create_pod` → `ssh_target` → `Tunnel` →
`ComfyUI.wait_ready`, storing `pod_id` and `started` via `write_state`).

```bash
nix develop -c comfy up
```

Expected output must include the chosen GPU, the pod id, `http://127.0.0.1:8188`, and
ComfyUI's version and torch build from `/system_stats`.

**This is the first live test of CUDA 13 template `wgd3p4n4o6`.** Record in the
completion notes: which GPU was allocated, ComfyUI version, torch version, time to
ready. If the Blackwell path fails, note exactly how, then re-run — `pick_gpu` should
fall through to the 4090, and if it does not, that is a bug to fix here.

- [ ] **Step 7: 🔴 LIVE — verify the tunnel really is the only path in**

With the pod still up, from another shell:
```bash
curl -sS -m 10 http://127.0.0.1:8188/system_stats | head -c 200; echo
POD=$(jq -r .pod_id .comfy-state.json)
curl -sS -m 15 -o /dev/null -w 'public 8188 -> %{http_code}\n' "https://$POD-8188.proxy.runpod.net/system_stats" || echo "public 8188 -> unreachable (correct)"
```
Expected: the local call returns JSON; the public proxy call returns **404 or fails**.
A `200` there means 8188 got exposed and the access requirement is violated — stop and
fix `create_pod`.

- [ ] **Step 8: 🔴 LIVE — tear down and confirm**

```bash
nix develop -c comfy down
nix develop -c comfy status
```
Expected: `down` prints the real spend from `/v2/billing/pods`; `status` then reports no
pod running. Confirm in the Runpod console that nothing is left.

- [ ] **Step 9: Commit**

```bash
git add src/comfy_runpod/tunnel.py src/comfy_runpod/comfyui.py src/comfy_runpod/cli.py tests/test_tunnel.py
git commit -m "feat(pod): up/down/status with SSH-only tunnel to ComfyUI"
```

---

### Task 6: Export the five workflows — live, human-in-the-loop, about $0.20

**⚠️ Needs a running pod and a human at a browser.** This is the step that avoids
writing a 1,760-line UI→API workflow converter.

**Files:**
- Create: `workflows/t2i.json`
- Create: `workflows/i2i.json`
- Create: `workflows/t2v.json`
- Create: `workflows/i2v.json`
- Create: `workflows/v2v.json`
- Create: `workflows/README.md`

**Interfaces:**
- Consumes: a running pod from Task 5.
- Produces: five API-format graphs. Every one MUST carry `_meta.title` on the nodes the
  harness patches, using exactly these titles: `positive`, `negative`, `latent`,
  `sampler`, `output`, and `input` for the three input-taking modes.

- [ ] **Step 1: Bring a pod up and open the browser**

```bash
nix develop -c comfy up
xdg-open http://127.0.0.1:8188
```

- [ ] **Step 2: For each of t2v, i2v, v2v — load the official template and export**

In ComfyUI: **Workflow → Browse Templates**, then load in turn:
- `Wan 2.2 5B Video Generation` → becomes `t2v.json` and `i2v.json` (TI2V 5B serves both;
  export twice, and in the `i2v` copy wire the `LoadImage` node in as the start frame)
- `Wan 2.2 14B Fun Control` → `v2v.json`, but **swap its model loader to
  `wan2.2_fun_control_5B_bf16.safetensors`** since that is what we downloaded

For each: rename the nodes to the canonical titles (double-click a node's title to edit)
— the positive prompt node to `positive`, negative to `negative`, the latent/size node to
`latent`, the sampler to `sampler`, the save node to `output`, and any `LoadImage` /
`LoadVideo` to `input`. Then **File → Export (API)** and save into `workflows/`.

- [ ] **Step 3: Build the t2i graph**

Load template `Z-Image: Text to Image` if present; otherwise build it: `UNETLoader`
(`z_image_bf16.safetensors`) → `CLIPTextEncode` ×2 → `EmptyLatentImage` →
`KSampler` → `VAEDecode` → `SaveImage`, with the `CLIPLoader` on
`qwen_3_4b.safetensors` and `VAELoader` on `ae.safetensors`.

Per the quality ruling, set `sampler` steps high rather than to a distilled minimum —
start at 30 steps, cfg 4.0, and confirm one image looks right before exporting. Apply the
canonical titles, then **File → Export (API)** to `workflows/t2i.json`.

- [ ] **Step 4: Build the i2i graph by hand**

There is no official classic-denoise template — the shipped "Image Edit" templates are
instruction-edit models, not a strength dial. Copy the t2i graph and rewire:

`LoadImage` (title `input`) → `VAEEncode` → `KSampler` (title `sampler`, `denoise: 0.6`)
→ `VAEDecode` → `SaveImage` (title `output`). Delete `EmptyLatentImage`; the
`VAEEncode` output now feeds `latent_image`. Keep both `CLIPTextEncode` nodes titled
`positive` and `negative`.

Export to `workflows/i2i.json`.

- [ ] **Step 5: Verify all five parse and carry the required titles**

```bash
nix develop -c python - <<'PY'
import json, pathlib, sys
REQUIRED = {"positive", "negative", "sampler", "output"}
NEEDS_INPUT = {"i2i", "i2v", "v2v"}
bad = False
for f in sorted(pathlib.Path("workflows").glob("*.json")):
    g = json.loads(f.read_text())
    titles = {n.get("_meta", {}).get("title") for n in g.values()}
    missing = REQUIRED - titles
    if f.stem in NEEDS_INPUT and "input" not in titles:
        missing |= {"input"}
    print(f"{f.name}: {len(g)} nodes, missing={sorted(missing) or 'none'}")
    bad |= bool(missing)
    assert "definitions" not in g, f"{f.name} is UI format, not API format"
sys.exit(1 if bad else 0)
PY
```
Expected: five lines, every one `missing=none`, exit 0. A `definitions` key means the
export was UI format — redo it via **Export (API)**, not plain Export.

- [ ] **Step 6: Generate one output per mode in the browser to prove the models work**

Queue each workflow once from the UI. All five must produce a file. This is where a
missing or misnamed model file surfaces — cheaper to find now than inside `comfy run`.

While here, settle spec §13.4: render the same `t2v` prompt with
`umt5_xxl_fp8_e4m3fn_scaled` and note whether it is visibly worse than expected. If you
cannot tell, keep fp8 and save the $0.33/month.

- [ ] **Step 7: Tear down**

```bash
nix develop -c comfy down
```

- [ ] **Step 8: Write `workflows/README.md`**

```markdown
# Workflows

API-format ComfyUI graphs, one per mode. Exported from a live ComfyUI via
**File → Export (API)** — not plain Export, which produces UI format with a
`definitions` key that the harness cannot read.

The harness finds nodes by `_meta.title`, so these titles are load-bearing:

| Title | Node | Patched with |
| --- | --- | --- |
| `positive` | CLIPTextEncode | `prompt` |
| `negative` | CLIPTextEncode | `negative` |
| `latent` | EmptyLatentImage (t2i/t2v only) | `size` |
| `sampler` | KSampler | `seed`, plus any `sampler.*` override |
| `output` | SaveImage / SaveVideo | — |
| `input` | LoadImage / LoadVideo (i2i, i2v, v2v) | uploaded `input` file |

Renaming a node in the UI breaks the harness. Re-export after any edit, and
re-run the title check in Task 6 Step 5 of the implementation plan.
```

- [ ] **Step 9: Commit**

```bash
git add workflows/
git commit -m "feat(workflows): API-format graphs for all five modes"
```

---

### Task 7: Workflow parameterisation

Pure logic, no cloud, no money. The most heavily tested module because it is the only one
with real branching.

**Files:**
- Create: `src/comfy_runpod/workflow.py`
- Create: `tests/test_workflow.py`

**Interfaces:**
- Consumes: `config.Run`.
- Produces:
  - `WorkflowError(Exception)`.
  - `load(mode: str, directory: Path) -> dict` — reads `<directory>/<mode>.json`.
  - `find_by_title(graph: dict, title: str) -> str` — returns the node id; raises if absent or ambiguous.
  - `apply_run(graph: dict, run: Run, seed: int, uploaded_name: str | None = None) -> dict` — returns a **new** graph; never mutates the input.
  - `apply_overrides(graph: dict, overrides: dict) -> dict` — `"sampler.cfg": 3.5` sets `inputs["cfg"]` on the node titled `sampler`.
  - `seeds_for(run: Run) -> list[int]` — `count` distinct seeds; consecutive from `run.seed` when fixed, random when `None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_workflow.py`:

```python
import copy
import json

import pytest

from comfy_runpod.config import Run
from comfy_runpod.workflow import (
    WorkflowError,
    apply_overrides,
    apply_run,
    find_by_title,
    load,
    seeds_for,
)

GRAPH = {
    "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"},
          "_meta": {"title": "positive"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "old neg"},
          "_meta": {"title": "negative"}},
    "3": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 512, "height": 512, "batch_size": 1},
          "_meta": {"title": "latent"}},
    "4": {"class_type": "KSampler",
          "inputs": {"seed": 0, "steps": 30, "cfg": 4.0},
          "_meta": {"title": "sampler"}},
    "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x"},
          "_meta": {"title": "output"}},
}


def run(**kw) -> Run:
    base = dict(mode="t2i", prompt="a fox", negative="blurry",
                count=2, size=(1024, 768))
    base.update(kw)
    return Run(**base)


def test_find_by_title():
    assert find_by_title(GRAPH, "sampler") == "4"


def test_find_by_title_missing_raises_with_available_titles():
    with pytest.raises(WorkflowError, match="nope"):
        find_by_title(GRAPH, "nope")


def test_find_by_title_ambiguous_raises():
    g = copy.deepcopy(GRAPH)
    g["9"] = {"class_type": "KSampler", "inputs": {}, "_meta": {"title": "sampler"}}
    with pytest.raises(WorkflowError, match="more than one"):
        find_by_title(g, "sampler")


def test_apply_run_sets_prompt_negative_size_and_seed():
    out = apply_run(GRAPH, run(), seed=1234)
    assert out["1"]["inputs"]["text"] == "a fox"
    assert out["2"]["inputs"]["text"] == "blurry"
    assert out["3"]["inputs"]["width"] == 1024
    assert out["3"]["inputs"]["height"] == 768
    assert out["4"]["inputs"]["seed"] == 1234


def test_apply_run_does_not_mutate_input():
    before = json.dumps(GRAPH, sort_keys=True)
    apply_run(GRAPH, run(), seed=7)
    assert json.dumps(GRAPH, sort_keys=True) == before


def test_apply_run_tolerates_missing_latent_node():
    """i2i has no EmptyLatentImage — size comes from the input image."""
    g = {k: v for k, v in copy.deepcopy(GRAPH).items() if k != "3"}
    out = apply_run(g, run(mode="i2i"), seed=5)
    assert out["4"]["inputs"]["seed"] == 5


def test_apply_run_sets_uploaded_image_on_input_node():
    g = copy.deepcopy(GRAPH)
    g["6"] = {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"},
              "_meta": {"title": "input"}}
    out = apply_run(g, run(mode="i2i"), seed=1, uploaded_name="cat_01.png")
    assert out["6"]["inputs"]["image"] == "cat_01.png"


def test_apply_run_requires_input_node_when_uploaded_name_given():
    with pytest.raises(WorkflowError, match="input"):
        apply_run(GRAPH, run(mode="i2i"), seed=1, uploaded_name="cat.png")


def test_apply_overrides_sets_nested_input():
    out = apply_overrides(GRAPH, {"sampler.cfg": 3.5})
    assert out["4"]["inputs"]["cfg"] == 3.5


def test_apply_overrides_can_add_an_input_not_already_present():
    out = apply_overrides(GRAPH, {"sampler.denoise": 0.6})
    assert out["4"]["inputs"]["denoise"] == 0.6


def test_apply_overrides_rejects_malformed_key():
    with pytest.raises(WorkflowError, match="title.input"):
        apply_overrides(GRAPH, {"cfg": 3.5})


def test_apply_overrides_rejects_unknown_title():
    with pytest.raises(WorkflowError, match="ghost"):
        apply_overrides(GRAPH, {"ghost.cfg": 1})


def test_seeds_for_fixed_seed_is_consecutive_and_reproducible():
    assert seeds_for(run(count=3, seed=100)) == [100, 101, 102]


def test_seeds_for_random_seed_is_distinct():
    seeds = seeds_for(run(count=20, seed=None))
    assert len(seeds) == 20
    assert len(set(seeds)) == 20
    assert all(0 <= s < 2**32 for s in seeds)


def test_load_reads_named_mode(tmp_path):
    (tmp_path / "t2i.json").write_text(json.dumps(GRAPH))
    assert load("t2i", tmp_path)["4"]["class_type"] == "KSampler"


def test_load_missing_mode_raises(tmp_path):
    with pytest.raises(WorkflowError, match="t2v.json"):
        load("t2v", tmp_path)


def test_load_rejects_ui_format(tmp_path):
    (tmp_path / "t2i.json").write_text(json.dumps({"nodes": [], "definitions": {}}))
    with pytest.raises(WorkflowError, match="API format"):
        load("t2i", tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nix develop -c python -m pytest tests/test_workflow.py -q`
Expected: `ModuleNotFoundError: No module named 'comfy_runpod.workflow'`.

- [ ] **Step 3: Write `src/comfy_runpod/workflow.py`**

```python
"""Patch committed API-format graphs. Nodes are addressed by _meta.title.

This is the whole reason the config's `overrides` escape hatch costs a dict
merge rather than a feature: a run's named fields and a user's raw override
both resolve through the same title lookup.
"""

from __future__ import annotations

import copy
import json
import secrets
from pathlib import Path

from .config import Run


class WorkflowError(Exception):
    """A graph is missing, the wrong format, or lacks a node we must patch."""


def load(mode: str, directory: Path) -> dict:
    path = Path(directory) / f"{mode}.json"
    if not path.is_file():
        raise WorkflowError(f"{path} not found — export it per the plan's Task 6")
    try:
        graph = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise WorkflowError(f"{path} is not valid JSON: {e}") from e
    if "definitions" in graph or "nodes" in graph:
        raise WorkflowError(
            f"{path} is UI format, not API format. Re-export with "
            "File -> Export (API)."
        )
    return graph


def find_by_title(graph: dict, title: str) -> str:
    hits = [
        nid for nid, node in graph.items()
        if isinstance(node, dict) and node.get("_meta", {}).get("title") == title
    ]
    if len(hits) > 1:
        raise WorkflowError(
            f"more than one node is titled {title!r} ({', '.join(hits)}) — "
            "titles must be unique"
        )
    if not hits:
        available = sorted(
            t for t in (
                n.get("_meta", {}).get("title")
                for n in graph.values() if isinstance(n, dict)
            ) if t
        )
        raise WorkflowError(
            f"no node titled {title!r}. Available: {', '.join(available) or 'none'}"
        )
    return hits[0]


def _set(graph: dict, title: str, key: str, value: object) -> None:
    graph[find_by_title(graph, title)]["inputs"][key] = value


def _set_if_present(graph: dict, title: str, key: str, value: object) -> None:
    try:
        nid = find_by_title(graph, title)
    except WorkflowError:
        return
    graph[nid]["inputs"][key] = value


def apply_run(
    graph: dict, run: Run, seed: int, uploaded_name: str | None = None
) -> dict:
    out = copy.deepcopy(graph)
    _set(out, "positive", "text", run.prompt)
    _set_if_present(out, "negative", "text", run.negative)
    # i2i has no EmptyLatentImage: its size comes from the input image.
    _set_if_present(out, "latent", "width", run.size[0])
    _set_if_present(out, "latent", "height", run.size[1])
    _set(out, "sampler", "seed", seed)
    if uploaded_name is not None:
        _set(out, "input", "image", uploaded_name)
    return out


def apply_overrides(graph: dict, overrides: dict) -> dict:
    out = copy.deepcopy(graph)
    for key, value in overrides.items():
        if "." not in key:
            raise WorkflowError(
                f"override {key!r} must be written 'title.input', e.g. 'sampler.cfg'"
            )
        title, _, field = key.partition(".")
        out[find_by_title(out, title)]["inputs"][field] = value
    return out


def seeds_for(run: Run) -> list[int]:
    if run.seed is None:
        return [secrets.randbelow(2**32) for _ in range(run.count)]
    return [run.seed + i for i in range(run.count)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `nix develop -c python -m pytest tests/test_workflow.py -q`
Expected: all pass (17 tests).

- [ ] **Step 5: Run the whole suite**

Run: `nix develop -c python -m pytest tests/ -q`
Expected: all pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/comfy_runpod/workflow.py tests/test_workflow.py
git commit -m "feat(workflow): patch API graphs by node title, with overrides"
```

---

### Task 8: `comfy run` — the batch loop. Live verification, about $0.30

**Files:**
- Modify: `src/comfy_runpod/comfyui.py`
- Modify: `src/comfy_runpod/cli.py`
- Create: `tests/test_comfyui.py`

**Interfaces:**
- Consumes: `workflow.apply_run`/`apply_overrides`/`seeds_for`, `comfyui.ComfyUI`, `tunnel.Tunnel`.
- Produces:
  - `ComfyUI.upload_image(path: Path) -> str` — returns the server-side filename.
  - `ComfyUI.queue(graph: dict, client_id: str) -> str` — returns `prompt_id`.
  - `ComfyUI.history(prompt_id: str) -> dict | None` — `None` while still running.
  - `ComfyUI.outputs_of(prompt_id: str) -> list[dict]` — `{filename, subfolder, type}` entries.
  - `ComfyUI.download(entry: dict, dest_dir: Path) -> Path`.
  - `cli.cmd_run(args) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_comfyui.py`:

```python
import json

import pytest

from comfy_runpod.comfyui import ComfyUI, ComfyUIError


class FakeComfy(ComfyUI):
    """Overrides only the two transport seams, so URL/parse logic is real."""

    def __init__(self, responses):
        super().__init__("http://127.0.0.1:8188")
        self.responses = responses
        self.posted = []

    def _get_json(self, path, timeout=30):
        if path not in self.responses:
            raise ComfyUIError(f"no canned response for {path}")
        return self.responses[path]

    def _post_json(self, path, body, timeout=60):
        self.posted.append((path, body))
        return self.responses.get(path, {})


def test_queue_posts_prompt_with_client_id():
    c = FakeComfy({"/prompt": {"prompt_id": "p1"}})
    assert c.queue({"1": {}}, client_id="cid") == "p1"
    path, body = c.posted[-1]
    assert path == "/prompt"
    assert body["prompt"] == {"1": {}}
    assert body["client_id"] == "cid"


def test_queue_raises_on_validation_error():
    c = FakeComfy({"/prompt": {"error": {"message": "bad node"}}})
    with pytest.raises(ComfyUIError, match="bad node"):
        c.queue({"1": {}}, client_id="cid")


def test_history_returns_none_while_pending():
    c = FakeComfy({"/history/p1": {}})
    assert c.history("p1") is None


def test_history_returns_entry_when_complete():
    c = FakeComfy({"/history/p1": {"p1": {"outputs": {"5": {"images": []}}}}})
    assert c.history("p1") == {"outputs": {"5": {"images": []}}}


def test_outputs_of_collects_images_and_videos():
    hist = {
        "p1": {
            "outputs": {
                "5": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
                "6": {"videos": [{"filename": "b.mp4", "subfolder": "v", "type": "output"}]},
            }
        }
    }
    c = FakeComfy({"/history/p1": hist})
    names = sorted(e["filename"] for e in c.outputs_of("p1"))
    assert names == ["a.png", "b.mp4"]


def test_outputs_of_ignores_non_file_output_keys():
    hist = {"p1": {"outputs": {"7": {"text": ["hello"]}}}}
    c = FakeComfy({"/history/p1": hist})
    assert c.outputs_of("p1") == []


def test_view_url_encodes_subfolder_and_type():
    c = ComfyUI("http://127.0.0.1:8188")
    url = c.view_url({"filename": "a b.png", "subfolder": "sub dir", "type": "output"})
    assert "filename=a+b.png" in url or "filename=a%20b.png" in url
    assert "subfolder=sub+dir" in url or "subfolder=sub%20dir" in url
    assert "type=output" in url
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nix develop -c python -m pytest tests/test_comfyui.py -q`
Expected: failures — `_post_json`, `queue`, `history`, `outputs_of`, `view_url` do not exist.

- [ ] **Step 3: Extend `src/comfy_runpod/comfyui.py`**

Append to the existing class:

```python
    # --- transport seams -------------------------------------------------

    def _post_json(self, path: str, body: dict, timeout: int = 60) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
        except (urllib.error.URLError, OSError) as e:
            raise ComfyUIError(f"POST {path} failed: {e}") from e
        return json.loads(raw) if raw else {}

    # --- queueing --------------------------------------------------------

    def queue(self, graph: dict, client_id: str) -> str:
        out = self._post_json("/prompt", {"prompt": graph, "client_id": client_id})
        if "error" in out:
            err = out["error"]
            msg = err.get("message", err) if isinstance(err, dict) else err
            raise ComfyUIError(f"ComfyUI rejected the graph: {msg}")
        pid = out.get("prompt_id")
        if not pid:
            raise ComfyUIError(f"/prompt returned no prompt_id: {out}")
        return str(pid)

    def history(self, prompt_id: str) -> dict | None:
        out = self._get_json(f"/history/{prompt_id}")
        return out.get(prompt_id)

    def outputs_of(self, prompt_id: str) -> list[dict]:
        entry = self.history(prompt_id) or {}
        found: list[dict] = []
        for node_output in (entry.get("outputs") or {}).values():
            for kind in ("images", "videos", "gifs", "audio"):
                found.extend(node_output.get(kind) or [])
        return found

    # --- files -----------------------------------------------------------

    def upload_image(self, path: Path) -> str:
        """Multipart by hand — one stdlib-only function beats adding a dependency."""
        boundary = f"----comfy{secrets.token_hex(8)}"
        payload = b"".join([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; '
            f'filename="{path.name}"\r\n'.encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            path.read_bytes(),
            f"\r\n--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
            f"--{boundary}--\r\n".encode(),
        ])
        req = urllib.request.Request(
            f"{self.base}/upload/image",
            data=payload,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise ComfyUIError(f"uploading {path.name} failed: {e}") from e
        name = out.get("name")
        if not name:
            raise ComfyUIError(f"/upload/image returned no name: {out}")
        if out.get("subfolder"):
            name = f"{out['subfolder']}/{name}"
        return str(name)

    def view_url(self, entry: dict) -> str:
        q = urllib.parse.urlencode({
            "filename": entry.get("filename", ""),
            "subfolder": entry.get("subfolder", ""),
            "type": entry.get("type", "output"),
        })
        return f"{self.base}/view?{q}"

    def download(self, entry: dict, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / entry["filename"]
        try:
            with urllib.request.urlopen(self.view_url(entry), timeout=600) as r, \
                 dest.open("wb") as fh:
                shutil.copyfileobj(r, fh)
        except (urllib.error.URLError, OSError) as e:
            raise ComfyUIError(f"downloading {entry['filename']} failed: {e}") from e
        return dest
```

Add to the imports at the top of the file: `import secrets`, `import shutil`,
`import urllib.parse`, and `from pathlib import Path`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `nix develop -c python -m pytest tests/test_comfyui.py -q`
Expected: all pass (7 tests).

- [ ] **Step 5: Write `cmd_run` in `cli.py`**

```python
import datetime as _dt
import re
import secrets
import time

from . import workflow
from .comfyui import ComfyUI, ComfyUIError
from .config import load_run, read_state, write_state, clear_state
from .tunnel import Tunnel

WORKFLOW_DIR = Path("workflows")


def _slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:limit] or "run"


def cmd_run(args) -> int:
    infra = load_infra(INFRA_PATH)
    run = load_run(Path(args.run_file))
    graph = workflow.load(run.mode, WORKFLOW_DIR)
    if run.overrides:
        graph = workflow.apply_overrides(graph, run.overrides)

    client = _client()
    state = read_state()
    started_here = not state.get("pod_id")
    pod_id = state.get("pod_id") or _bring_up(client, infra)

    host, port, user = client.ssh_target(pod_id)
    out_dir = Path("out") / f"{_dt.date.today().isoformat()}-{_slug(run.prompt)}"
    client_id = secrets.token_hex(8)

    try:
        with Tunnel(host, port, user) as tun:
            comfy = ComfyUI(tun.url)
            comfy.wait_ready()

            uploaded = comfy.upload_image(run.input) if run.input else None
            seeds = workflow.seeds_for(run)

            prompt_ids = []
            for seed in seeds:
                g = workflow.apply_run(graph, run, seed, uploaded)
                prompt_ids.append(comfy.queue(g, client_id))
            print(f"queued {len(prompt_ids)} job(s) -> {out_dir}")

            done = 0
            for pid in prompt_ids:
                while comfy.history(pid) is None:
                    time.sleep(2)
                for entry in comfy.outputs_of(pid):
                    comfy.download(entry, out_dir)
                done += 1
                print(f"  {done}/{len(prompt_ids)}")
    finally:
        if not args.keep:
            spend = client.pod_spend(pod_id)
            client.terminate_pod(pod_id)
            clear_state()
            cost = f"${spend:.2f}" if spend is not None else "unknown"
            print(f"terminated {pod_id} — spend {cost}")
        elif started_here:
            print(f"pod {pod_id} left running (--keep). `comfy down` when finished.")

    print(f"saved to {out_dir}")
    return 0
```

`_bring_up` is the same helper `cmd_up` uses — extract it in Task 5 so both share it.

- [ ] **Step 6: Verify a bad run file fails before any pod is created**

Run:
```bash
printf 'mode: t2x\nprompt: x\ncount: 1\n' > /tmp/bad.yaml
nix develop -c comfy run /tmp/bad.yaml; echo "exit=$?"
```
Expected: `error: ... mode 't2x' is not one of ...`, `exit=1`, and **no pod created**.
Validation must precede provisioning or a typo costs money. Confirm with
`comfy status`.

- [ ] **Step 7: 🔴 LIVE — a real t2i batch (~$0.10)**

```bash
mkdir -p prompts
cat > prompts/fox.yaml <<'YAML'
mode: t2i
prompt: "a red fox in a misty pine forest, cinematic, shallow depth of field"
negative: "blurry, watermark, text"
count: 4
size: [1024, 1024]
seed: random
YAML
nix develop -c comfy run prompts/fox.yaml
```
Expected: pod boots, 4 images land in `out/<today>-a-red-fox-in-a-misty.../`, pod
terminates, real spend printed. Open the images and confirm they are not noise.

- [ ] **Step 8: 🔴 LIVE — one video and one input-taking mode (~$0.20)**

```bash
cat > prompts/clip.yaml <<'YAML'
mode: i2v
prompt: "the fox turns its head slowly toward the camera"
count: 1
size: [1280, 704]
input: ../out/<pick-one>.png
YAML
nix develop -c comfy run prompts/clip.yaml
```
Expected: the image uploads, one video file downloads, pod terminates. This exercises
`upload_image`, the `input` node title, and a long `/view` download — the three things
that only fail on real hardware.

- [ ] **Step 9: Commit**

```bash
git add src/comfy_runpod/comfyui.py src/comfy_runpod/cli.py tests/test_comfyui.py prompts/
git commit -m "feat(run): queue batches, stream progress, download outputs"
```

---

### Task 9: README, full verification, and the required quality passes

**Files:**
- Create: `README.md`
- Modify: whatever the quality passes flag.

- [ ] **Step 1: Write `README.md`**

Cover: what it is; `nix develop`; `RUNPOD_API_KEY` setup; `cp comfy.example.yaml
comfy.yaml`; `comfy provision` once and paste the volume id; the run-file format with a
worked example per mode; the cost model (**$5.25/month standing for the volume**,
~$0.72/hr while a pod runs, `comfy run` terminates by default); and the plain warning
that **v2 has no server-side pod TTL**, so `--keep` plus a crashed laptop bills ~$17/day
until noticed.

Also record the **measured download throughput** from Task 4 Step 7 — it is the number
that justified buying a volume, and the next person will want it.

Mention the zero-setup alternative for plain t2i/t2v that we deliberately did not wire
in: Runpod Public Endpoints, about $0.005/image and $0.30/clip, reachable with `curl`.

- [ ] **Step 2: Run the full suite**

Run: `nix develop -c python -m pytest tests/ -q`
Expected: every test passes. Paste the real count into the completion notes.

- [ ] **Step 3: Verify the flake builds from a clean checkout**

Run: `nix build .# && ./result/bin/comfy --help`
Expected: usage block, exit 0. This catches a `pyproject.toml` that only works because
`PYTHONPATH` was set in the dev shell.

- [ ] **Step 4: Confirm no paid resource is left running**

Run: `nix develop -c comfy status`
Expected: no pod running. Then check the Runpod console for stray pods, and confirm the
volume is the only standing charge.

- [ ] **Step 5: Run `/simplify`**

Required by CLAUDE.md. Apply what it finds.

- [ ] **Step 6: Run `/ponytail:ponytail-review`**

Required by CLAUDE.md. Apply what it finds.

- [ ] **Step 7: Commit and merge to `develop`**

```bash
git add -A
git commit -m "docs: README with setup, costs and measured throughput"
```

Merge the feature branch with `--no-ff` per CLAUDE.md gitflow.

---

### Task 10: Teardown — leave nothing billing

The user asked that no recurring charge survive this session. Terminating a pod is routine;
**deleting the volume destroys 48.7 GB of downloaded models irreversibly, so it stops and
asks** rather than being done automatically.

**Files:**
- Modify: `src/comfy_runpod/cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `runpod_api.Client`.
- Produces: `Client.list_pods() -> list[dict]`, `Client.list_volumes() -> list[dict]`, `Client.delete_volume(volume_id: str) -> None`, and a `comfy teardown` subcommand.

- [ ] **Step 1: Add the three client methods**

```python
    def list_pods(self) -> list[dict]:
        return list(self._call("GET", "/pods").get("pods") or [])

    def list_volumes(self) -> list[dict]:
        out = self._call("GET", "/network-volumes")
        return list(out.get("networkVolumes") or out.get("volumes") or [])

    def delete_volume(self, volume_id: str) -> None:
        """Irreversible. Every model on the volume is lost."""
        self._call("DELETE", f"/network-volumes/{volume_id}")
```

`_http` already takes a method string, so `DELETE` needs no new plumbing. If the live
response nests volumes under a different key, fix `list_volumes` to match what the API
actually returns and note it in the report.

- [ ] **Step 2: Add `comfy teardown`**

Register a `teardown` subparser with `--delete-volume` (default `False`), and a handler
that terminates every running pod unconditionally, then reports each volume with its size
and monthly cost. It deletes a volume **only** when `--delete-volume` is passed, and
prints the consequence first:

```python
def cmd_teardown(args) -> int:
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
    for v in volumes:
        size = v.get("size", 0)
        print(f"volume {v.get('id')}  {size} GB  ~${size * 0.07:.2f}/month")
    if not args.delete_volume:
        print("\nVolumes still bill monthly. `comfy teardown --delete-volume` removes "
              "them — that destroys every downloaded model and cannot be undone.")
        return 0
    for v in volumes:
        print(f"deleting {v.get('id')} — models are gone")
        client.delete_volume(str(v["id"]))
    return 0
```

- [ ] **Step 3: Verify pods are gone without deleting anything**

Run: `nix develop -c comfy teardown`
Expected: every pod terminated, each volume listed with its monthly cost, and **no
deletion**. Exit 0.

- [ ] **Step 4: 🛑 STOP — ask before deleting the volume**

Deleting the volume is irreversible and forfeits ~$0.15 of download plus 54 minutes of
wall clock to redo. Present the choice and wait:

- **Keep it** — $5.25/month standing, every future `comfy up` ready in ~3 min.
- **Delete it** — $0/month, and the next session re-runs `comfy provision`.

Run `comfy teardown --delete-volume` only on an explicit yes.

- [ ] **Step 5: Confirm the end state**

```bash
nix develop -c comfy teardown
```
Expected: `no pods running` and either the volume listed, or `no volumes — nothing is
billing`. Whichever it is, state it plainly in the final report.

- [ ] **Step 6: Commit**

```bash
git add src/comfy_runpod/cli.py README.md
git commit -m "feat(cli): teardown command to stop all billing"
```

---

## Volume-optional operation

`create_pod` accepts `volume_id=None` and omits `mounts`. This is load-bearing, not
defensive: the session is expected to end with the volume deleted, and a harness that
could only run *with* a volume would be unusable in exactly that state. A volume-less run
re-downloads models on each boot — slower and, on a bad host, much slower, but it works
and costs nothing while idle.

## Self-Review

**Spec coverage:**

| Spec section | Task |
| --- | --- |
| §1 goal, five modes | 6, 7, 8 |
| §2 decisions | throughout; config shape in 2, overrides in 7 |
| §3 volume, 75 GB, EU-RO-1 | 4 |
| §4 REST v2 only | 3 |
| §5 placement, GPU order, template pairing | 2 (config), 3 (`pick_gpu`), 5 (live) |
| §6 model set, 48.7 GB | 4 |
| §7 zero custom nodes, API-format export, `_meta.title` | 6, 7 |
| §8 config schema | 2 |
| §9 five commands | 4 (`provision`), 5 (`up`/`down`/`status`), 8 (`run`) |
| §10 SSH-only, cost guards | 5 (tunnel + exposure check), 8 (terminate-on-completion) |
| §11 flake | 1, verified again in 9 |
| §12 verification | 4.7, 5.6, 5.7, 6.6, 8.7, 8.8, 9 |
| §13.4 umt5 fp8 vs fp16 | 6.6 |
| User request: no recurring charge left at the end | 10 |
| Volume-optional operation (follows from Task 10) | 3, and the note above |

No gaps found.

**Placeholder scan:** one intentional `<pick-one>` in Task 8 Step 8 — the input image
cannot be named before Task 8 Step 7 produces it. Every other step carries runnable
content.

**Type consistency check:** `GpuChoice`/`Infra`/`Run` defined in Task 2 and consumed
unchanged in 3, 4, 7, 8. `Client.ssh_target` returns `(host, port, user)` in Task 3 and is
unpacked that way in 4, 5, 8. `ComfyUI._get_json`/`_post_json` are the seams the Task 8
fake overrides, and both exist by then (`_get_json` from Task 5). `find_by_title` is used
by `apply_run` and `apply_overrides` in the same module. `_bring_up` is shared between
`cmd_up` and `cmd_run` — Task 5 must extract it, which Task 8 Step 5 restates.

**Known cross-task dependency to watch:** Task 5 Step 6 wires `up`/`down`/`status` into
the CLI but the plan shows the dispatch table only in Task 4 Step 5. Whoever executes
Task 5 must extend that same `handlers` dict rather than rewriting `main`.

## Cost Summary

| Task | Cost | Standing |
| --- | --- | --- |
| 1, 2, 3, 7 | $0 | — |
| 4 provision | ~$0.15 | **+$5.25/month volume** |
| 5 pod boot | ~$0.10 | — |
| 6 workflow export | ~$0.20 | — |
| 8 run verification | ~$0.30 | — |
| 10 teardown | $0 | **$0 if volume deleted** |
| **Total to working** | **~$0.75** | **$5.25/month, or $0 after teardown** |
