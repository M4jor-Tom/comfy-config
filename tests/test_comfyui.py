import json
import urllib.error

import pytest

import comfy_runpod.comfyui as comfyui_mod
from comfy_runpod.comfyui import ComfyUI, ComfyUIError


class FakeResponse:
    """Stands in for the context manager urllib.request.urlopen returns."""

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_system_stats_returns_parsed_json(monkeypatch):
    monkeypatch.setattr(
        comfyui_mod.urllib.request,
        "urlopen",
        lambda url, timeout=None: FakeResponse({"system": {"comfyui_version": "0.3.0"}}),
    )
    stats = ComfyUI("http://127.0.0.1:8188").system_stats()
    assert stats["system"]["comfyui_version"] == "0.3.0"


def test_system_stats_wraps_unreachable_host_as_comfyui_error(monkeypatch):
    def boom(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(comfyui_mod.urllib.request, "urlopen", boom)
    with pytest.raises(ComfyUIError, match="system_stats"):
        ComfyUI("http://127.0.0.1:8188").system_stats()


def test_wait_ready_succeeds_immediately_when_already_up(monkeypatch):
    monkeypatch.setattr(
        comfyui_mod.urllib.request,
        "urlopen",
        lambda url, timeout=None: FakeResponse({"system": {}}),
    )
    stats = ComfyUI("http://127.0.0.1:8188").wait_ready(timeout=5)
    assert stats == {"system": {}}


def test_wait_ready_retries_past_early_failures_then_succeeds(monkeypatch):
    """First boot is documented as 404 -> 502 -> 200: wait_ready must poll
    through failures rather than treating the first one as final."""
    calls = {"n": 0}

    def flaky(url, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise urllib.error.URLError("not ready yet")
        return FakeResponse({"system": {"comfyui_version": "0.3.0"}})

    monkeypatch.setattr(comfyui_mod.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(comfyui_mod.time, "sleep", lambda s: None)  # skip the real 10s waits

    stats = ComfyUI("http://127.0.0.1:8188").wait_ready(timeout=60)

    assert calls["n"] == 3
    assert stats["system"]["comfyui_version"] == "0.3.0"


def test_wait_ready_raises_comfyui_error_after_timeout(monkeypatch):
    calls = {"n": 0}

    def always_fails(url, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("still not ready")

    monkeypatch.setattr(comfyui_mod.urllib.request, "urlopen", always_fails)

    # timeout=0: the deadline is already in the past by the first loop check,
    # so this raises immediately with no real wait and no polling attempts.
    with pytest.raises(ComfyUIError, match="not ready within 0s"):
        ComfyUI("http://127.0.0.1:8188").wait_ready(timeout=0)
    assert calls["n"] == 0
