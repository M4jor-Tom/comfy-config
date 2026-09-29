"""One-time: create the volume and fill it with models via a cheap CPU pod.

A CPU pod costs $0.035/vCPU/hr against $0.72/hr for the GPU, and network
volumes attach to CPU pods. So a slow download host costs pennies instead of
dollars — even a 9-hour worst case is about $0.63, once.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .config import Infra
from .runpod_api import Client, RunpodError

MODELS_ROOT = "/workspace/runpod-slim/ComfyUI/models"

# 48.7 GB of models plus the ComfyUI install ComfyUI copies onto /workspace.
VOLUME_SIZE_GB = 75

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
    """Bash that runs on the pod. curl only — no Python, no hf CLI needed.

    The pod image (runpod/base) is a bare Ubuntu base and cannot be assumed to
    have Python installed. So everything that the original design computed
    on-pod with `python3 -c` is instead pre-computed here, at render time:
    per-file minimum-byte thresholds become integer literals, and the closing
    throughput figure is computed on-pod with plain bash integer arithmetic
    (fixed-point, one decimal place) rather than awk/bc/python.
    """
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"ROOT={MODELS_ROOT}",
        'mkdir -p "$ROOT"/{diffusion_models,text_encoders,vae,loras}',
        "",
        "started=$(date +%s)",
        "total_bytes=0",
        "",
    ]
    for m in MODEL_FILES:
        name = m.path.rsplit("/", 1)[-1]
        url = f"https://huggingface.co/{m.repo}/resolve/main/{m.path}"
        dest = f'"$ROOT/{m.target_dir}/{name}"'
        min_bytes = int(m.gb * 1e9 * 0.95)
        lines += [
            f"echo '==> {name} ({m.gb} GB)'",
            # --continue-at - resumes a partial file, so a retry is cheap.
            f"curl -fL --retry 5 --retry-delay 5 --continue-at - "
            f"-o {dest} '{url}'",
            # Trust measured bytes, not curl's exit code.
            f"actual=$(stat -c %s {dest})",
            f'if [ "$actual" -lt {min_bytes} ]; then',
            f'  echo "FAIL: {name} is $actual bytes, expected >= {min_bytes}" >&2; exit 1',
            "fi",
            "total_bytes=$(( total_bytes + actual ))",
            "",
        ]
    lines += [
        "elapsed=$(( $(date +%s) - started ))",
        'if [ "$elapsed" -le 0 ]; then',
        "  elapsed=1",
        "fi",
        # One-decimal MB/s via fixed-point integer math: no awk/bc/python needed.
        "mbps_tenths=$(( total_bytes * 10 / (1000000 * elapsed) ))",
        'echo "=== done: $(( total_bytes / 1000000 )) MB in ${elapsed}s '
        '($(( mbps_tenths / 10 )).$(( mbps_tenths % 10 )) MB/s) ==="',
        f"du -sh {MODELS_ROOT}",
    ]
    return "\n".join(lines) + "\n"


def provision(client: Client, infra: Infra) -> str:
    """Create the volume if absent, then fill it from a CPU pod. Returns volume id."""
    volume_id = infra.volume_id
    if volume_id:
        print(f"reusing existing volume {volume_id}")
    else:
        # A create_volume() POST that lands server-side but whose response never
        # arrives (timeout/URLError) leaves an unrecorded volume that costs
        # $5.25/month. Look before creating so a retry after that finds it instead
        # of doubling it.
        matches = [
            v for v in client.list_volumes()
            if v.get("name") == "comfy-models" and v.get("id")
        ]
        if len(matches) > 1:
            # This is the exact failure mode the lookup above exists to catch, so
            # it is the likely case here, not a hypothetical: make it visible
            # rather than silently picking one and letting both bill forever.
            print(
                f"WARNING: found {len(matches)} volumes named comfy-models, not 1 "
                "— each one bills you monthly, and `comfy teardown` lists them:"
            )
            for v in matches:
                print(f"  - {v['id']}  {v.get('size', '?')} GB")
            print("Proceeding with the first.")
        if matches:
            volume_id = str(matches[0]["id"])
            print(f"found existing volume {volume_id} named comfy-models — reusing it")
        else:
            volume_id = client.create_volume(
                "comfy-models", VOLUME_SIZE_GB, infra.datacenter
            )
            print(f"created volume {volume_id} ({VOLUME_SIZE_GB} GB) — PUT THIS IN comfy.yaml")

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
        host, port, user = client.wait_for_ssh(pod_id)
        print(f"downloading {total_gb()} GB — ssh {user}@{host} -p {port}")
        _run_download(host, port, user)
    finally:
        client.terminate_pod(pod_id)
        print(f"terminated {pod_id}")
    return volume_id


def _run_download(host: str, port: int, user: str) -> None:
    script = render_download_script()
    cmd = [
        "ssh", "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        f"{user}@{host}", "bash -s",
    ]
    result = subprocess.run(cmd, input=script, text=True)
    if result.returncode != 0:
        raise RunpodError(f"model download failed with exit {result.returncode}")
