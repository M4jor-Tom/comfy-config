import re

import pytest

from comfy_runpod.config import MODES, Infra
from comfy_runpod.runpod_api import RunpodError
import comfy_runpod.provision as provision_mod
from comfy_runpod.provision import MODEL_FILES, render_download_script, total_gb
from tests.conftest import FakeRunpodClient

_MODE_MODEL_SET = {
    "t2i": {"z_image_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors"},
    "i2i": {"z_image_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors"},
    "t2v": {
        "wan2.2_ti2v_5B_fp16.safetensors",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "wan2.2_vae.safetensors",
    },
    "i2v": {
        "wan2.2_ti2v_5B_fp16.safetensors",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "wan2.2_vae.safetensors",
    },
    "v2v": {
        "wan2.2_fun_control_5B_bf16.safetensors",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "wan2.2_vae.safetensors",
    },
}


def test_every_mode_has_a_complete_model_set_within_budget():
    assert total_gb() == pytest.approx(48.7, abs=0.2)
    assert set(_MODE_MODEL_SET) == set(MODES)
    available = {m.path.rsplit("/", 1)[-1] for m in MODEL_FILES}
    for mode, needed in _MODE_MODEL_SET.items():
        missing = needed - available
        assert not missing, f"{mode} is missing {sorted(missing)}"


def test_every_model_comes_from_a_comfy_org_mirror():
    """Comfy-Org mirrors need no HuggingFace token. Anything else would."""
    for m in MODEL_FILES:
        assert m.repo.startswith("Comfy-Org/"), m.repo


def test_quality_ruling_uses_non_distilled_bf16_z_image():
    paths = [m.path for m in MODEL_FILES]
    assert any("z_image_bf16" in p for p in paths)
    assert not any("turbo" in p for p in paths)
    assert not any("int8" in p for p in paths)


def test_target_dirs_map_each_file_to_its_correct_directory():
    """Not just 'is this a known ComfyUI model-type folder' but 'is THIS file in the
    right one' -- cross-checked against the model's own directory in the upstream
    split_files layout, so a diffusion model can't be silently filed under vae."""
    allowed = {"diffusion_models", "text_encoders", "vae", "loras"}
    for m in MODEL_FILES:
        assert m.target_dir in allowed
        upstream_dir = m.path.rsplit("/", 2)[-2]
        assert m.target_dir == upstream_dir, (
            f"{m.path}: target_dir={m.target_dir!r} but upstream dir is {upstream_dir!r}"
        )


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
    assert "stat" in s
    # Not just measured -- enforced: a short/truncated file must actually abort the
    # script, not merely get logged. Ties the size comparison directly to exit 1
    # so a regression that keeps `stat` but drops enforcement cannot pass silently.
    assert re.search(r'if \[ "\$actual" -lt \d+ \]; then\n\s*echo "FAIL:[^\n]*exit 1', s)


def _infra(volume_id=None):
    return Infra(datacenter="EU-RO-1", gpus=[], volume_id=volume_id)


def test_provision_reuses_infras_volume_id_without_creating(monkeypatch):
    monkeypatch.setattr(provision_mod, "_run_download", lambda host, port, user: None)
    client = FakeRunpodClient()
    result = provision_mod.provision(client, _infra(volume_id="vol-existing"))
    assert result == "vol-existing"
    assert "create_volume" not in client.calls
    assert "list_volumes" not in client.calls


def test_provision_finds_existing_volume_by_name_before_creating(monkeypatch):
    """R2 fix: a lost create_volume response must not cause a duplicate $5.25/mo
    volume -- provision must look for one named comfy-models before creating."""
    monkeypatch.setattr(provision_mod, "_run_download", lambda host, port, user: None)
    client = FakeRunpodClient(volumes=[{"id": "vol-found", "name": "comfy-models"}])
    result = provision_mod.provision(client, _infra(volume_id=None))
    assert result == "vol-found"
    assert "list_volumes" in client.calls
    assert "create_volume" not in client.calls


def test_provision_creates_a_volume_only_when_none_exists(monkeypatch):
    monkeypatch.setattr(provision_mod, "_run_download", lambda host, port, user: None)
    client = FakeRunpodClient(volumes=[{"id": "vol-other", "name": "unrelated"}])
    result = provision_mod.provision(client, _infra(volume_id=None))
    assert result == "vol-created"
    assert "list_volumes" in client.calls
    assert "create_volume" in client.calls


def test_provision_warns_but_proceeds_when_volume_name_is_ambiguous(monkeypatch, capsys):
    """A duplicate comfy-models volume is exactly the failure mode the round-1 fix
    guards against, so it is the likely case, not a hypothetical: it must be made
    visible (id + size of every match, monthly billing, `comfy teardown`), and the
    run must still proceed rather than fail."""
    monkeypatch.setattr(provision_mod, "_run_download", lambda host, port, user: None)
    client = FakeRunpodClient(
        volumes=[
            {"id": "vol-a", "name": "comfy-models", "size": 75},
            {"id": "vol-b", "name": "comfy-models", "size": 75},
        ]
    )
    result = provision_mod.provision(client, _infra(volume_id=None))
    assert result == "vol-a"  # proceeds with the first
    assert "create_volume" not in client.calls

    out = capsys.readouterr().out
    assert "vol-a" in out and "vol-b" in out
    assert "75" in out
    assert "monthly" in out.lower()
    assert "comfy teardown" in out


def test_provision_terminates_the_pod_it_created_on_success(monkeypatch):
    monkeypatch.setattr(provision_mod, "_run_download", lambda host, port, user: None)
    client = FakeRunpodClient(pod_id="pod-77")
    provision_mod.provision(client, _infra(volume_id="vol-existing"))
    assert client.terminated == ["pod-77"]


def test_provision_terminates_the_pod_even_when_download_fails(monkeypatch):
    def boom(host, port, user):
        raise RunpodError("download exploded")

    monkeypatch.setattr(provision_mod, "_run_download", boom)
    client = FakeRunpodClient(pod_id="pod-88")
    with pytest.raises(RunpodError, match="download exploded"):
        provision_mod.provision(client, _infra(volume_id="vol-existing"))
    assert client.terminated == ["pod-88"]
