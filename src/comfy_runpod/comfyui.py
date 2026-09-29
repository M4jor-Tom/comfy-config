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
