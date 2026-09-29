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
  - { id: "NVIDIA RTX PRO 4500 Blackwell", template: "2lv7ev3wfp" }
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
    assert infra.gpus[0].template == "2lv7ev3wfp"
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
