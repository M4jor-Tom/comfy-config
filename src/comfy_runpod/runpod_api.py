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
