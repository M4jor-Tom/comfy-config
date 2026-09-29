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
