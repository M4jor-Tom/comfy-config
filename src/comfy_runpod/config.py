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


def _duration_seconds(path: Path, value: object) -> int:
    """Accept 3h / 45m / 90s / bare seconds."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    mult = _UNITS.get(text[-1:], None)
    if mult is None:
        try:
            return int(text)
        except ValueError as e:
            raise ConfigError(f"{path}: terminate_after {value!r} is not a duration") from e
    try:
        return int(text[:-1]) * mult
    except ValueError as e:
        raise ConfigError(f"{path}: terminate_after {value!r} is not a duration") from e


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
            raise ConfigError(f"{path}: each gpus entry needs an 'id', got {entry!r}")
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
        terminate_after_seconds=_duration_seconds(path, raw.get("terminate_after", "3h")),
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
    try:
        width, height = int(size[0]), int(size[1])
    except (ValueError, TypeError) as e:
        raise ConfigError(f"{path}: size dimensions must be integers, got {size!r}") from e

    seed_raw = raw.get("seed", "random")
    if str(seed_raw) == "random":
        seed = None
    else:
        try:
            seed = int(seed_raw)
        except (ValueError, TypeError) as e:
            raise ConfigError(f"{path}: seed {seed_raw!r} is not an integer or 'random'") from e

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
