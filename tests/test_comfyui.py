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


# --- queue / history / outputs_of ------------------------------------------


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


def test_queue_error_includes_node_errors_when_present():
    """A bare 'Prompt outputs failed validation' names nothing actionable --
    node_errors is where ComfyUI actually says which node and why."""
    c = FakeComfy({
        "/prompt": {
            "error": {"message": "Prompt outputs failed validation"},
            "node_errors": {"7": {"errors": ["seed out of range"]}},
        }
    })
    with pytest.raises(ComfyUIError, match="7"):
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


# --- upload_image / download -------------------------------------------------


def test_upload_image_returns_server_reported_name(monkeypatch, tmp_path):
    """The server's reported name -- not the local filename -- is what a
    caller must patch into a graph (dedup can rename it)."""
    f = tmp_path / "cat.png"
    f.write_bytes(b"fake-image-bytes")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return FakeResponse({"name": "cat_00001_.png", "subfolder": "", "type": "input"})

    monkeypatch.setattr(comfyui_mod.urllib.request, "urlopen", fake_urlopen)

    name = ComfyUI("http://127.0.0.1:8188").upload_image(f)

    assert name == "cat_00001_.png"
    req = captured["req"]
    assert req.full_url == "http://127.0.0.1:8188/upload/image"
    assert req.get_method() == "POST"
    assert b"fake-image-bytes" in req.data
    assert b'name="image"' in req.data  # same field name works for video, verified live


def test_upload_image_prefixes_server_reported_subfolder(monkeypatch, tmp_path):
    """Confirms the /upload/image round-trip for a NON-image file (video),
    since LoadVideo's input is patched with this same method's return value."""
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"video-bytes")
    monkeypatch.setattr(
        comfyui_mod.urllib.request, "urlopen",
        lambda req, timeout=None: FakeResponse(
            {"name": "clip.mp4", "subfolder": "uploads", "type": "input"}
        ),
    )
    name = ComfyUI("http://127.0.0.1:8188").upload_image(f)
    assert name == "uploads/clip.mp4"


def test_upload_image_raises_when_server_returns_no_name(monkeypatch, tmp_path):
    f = tmp_path / "x.png"
    f.write_bytes(b"x")
    monkeypatch.setattr(
        comfyui_mod.urllib.request, "urlopen",
        lambda req, timeout=None: FakeResponse({}),
    )
    with pytest.raises(ComfyUIError, match="no name"):
        ComfyUI("http://127.0.0.1:8188").upload_image(f)


def test_download_writes_response_bytes_to_dest_dir(monkeypatch, tmp_path):
    class FakeBinaryResponse:
        def __init__(self, data):
            self._data = data

        def read(self, n=-1):
            data, self._data = self._data, b""
            return data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        comfyui_mod.urllib.request, "urlopen",
        lambda url, timeout=None: FakeBinaryResponse(b"the-actual-file-bytes"),
    )

    dest_dir = tmp_path / "out" / "nested"  # must be created, not pre-existing
    path = ComfyUI("http://127.0.0.1:8188").download(
        {"filename": "a.png", "subfolder": "", "type": "output"}, dest_dir
    )

    assert path == dest_dir / "a.png"
    assert path.read_bytes() == b"the-actual-file-bytes"
