"""ComfyUI HTTP API, reached through the SSH tunnel at 127.0.0.1."""

from __future__ import annotations

import json
import secrets
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


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

    # --- queueing ----------------------------------------------------------

    def queue(self, graph: dict, client_id: str) -> str:
        out = self._post_json("/prompt", {"prompt": graph, "client_id": client_id})
        if "error" in out:
            err = out["error"]
            msg = err.get("message", err) if isinstance(err, dict) else err
            if out.get("node_errors"):
                msg = f"{msg}: {out['node_errors']}"
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

    # --- files ---------------------------------------------------------------

    def upload_image(self, path: Path) -> str:
        """Upload a local file to ComfyUI's input store, returning the
        server-side name to patch into a graph -- which can differ from
        `path.name` (dedup, or a subfolder prefix), so callers must use the
        returned value rather than the local filename.

        Despite the name, `/upload/image` (multipart field "image") is also
        ComfyUI's endpoint for video uploads consumed by LoadVideo -- live-
        verified against this project's own committed workflows (v2v's
        input is an .mp4 uploaded this same way). There is no separate
        video-upload endpoint.
        """
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
