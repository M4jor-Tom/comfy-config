import urllib.request

import pytest

from comfy_runpod.config import GpuChoice
from comfy_runpod.runpod_api import Client, RunpodError, _http

BLACKWELL = GpuChoice(id="NVIDIA RTX PRO 4500 Blackwell", template="wgd3p4n4o6")
ADA = GpuChoice(id="NVIDIA GeForce RTX 4090", template="cw3nka7d08")

CATALOG = {
    "gpus": [
        {
            "id": "NVIDIA RTX PRO 4500 Blackwell",
            "memory": 32,
            "dataCenters": [{"id": "EU-RO-1", "availability": "MEDIUM"}],
        },
        {
            "id": "NVIDIA GeForce RTX 4090",
            "memory": 24,
            "dataCenters": [
                {"id": "EU-RO-1", "availability": "HIGH"},
                {"id": "US-IL-1", "availability": "LOW"},
            ],
        },
    ]
}


class FakeTransport:
    """Records calls and replays canned responses."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        for key, value in self.responses.items():
            if key in path:
                return value
        return {}


def test_http_sets_a_real_user_agent(monkeypatch):
    """Runpod's edge 403s with Cloudflare error 1010 on urllib's default
    (effectively absent) User-Agent; any real one clears it. Confirmed live
    2026-09-29. FakeTransport bypasses _http entirely, so this exercises it
    directly to keep the regression from becoming invisible again."""
    captured = {}

    class _Resp:
        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        captured["user_agent"] = req.get_header("User-agent")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _http("GET", "/pods", None, "fake-key")
    assert captured["user_agent"]
    assert "python-urllib" not in captured["user_agent"].lower()


def test_pick_gpu_prefers_first_with_capacity():
    t = FakeTransport({"/catalog/gpus": CATALOG})
    chosen, avail = Client("k", transport=t).pick_gpu([BLACKWELL, ADA], "EU-RO-1")
    assert chosen is BLACKWELL
    assert avail == "MEDIUM"


def test_pick_gpu_falls_through_when_primary_absent_from_datacenter():
    catalog = {
        "gpus": [
            {"id": BLACKWELL.id, "dataCenters": [{"id": "US-IL-1", "availability": "HIGH"}]},
            {"id": ADA.id, "dataCenters": [{"id": "EU-RO-1", "availability": "HIGH"}]},
        ]
    }
    t = FakeTransport({"/catalog/gpus": catalog})
    chosen, _ = Client("k", transport=t).pick_gpu([BLACKWELL, ADA], "EU-RO-1")
    assert chosen is ADA


def test_pick_gpu_skips_none_availability():
    catalog = {
        "gpus": [
            {"id": BLACKWELL.id, "dataCenters": [{"id": "EU-RO-1", "availability": "NONE"}]},
            {"id": ADA.id, "dataCenters": [{"id": "EU-RO-1", "availability": "LOW"}]},
        ]
    }
    t = FakeTransport({"/catalog/gpus": catalog})
    chosen, avail = Client("k", transport=t).pick_gpu([BLACKWELL, ADA], "EU-RO-1")
    assert chosen is ADA
    assert avail == "LOW"


def test_pick_gpu_raises_when_nothing_available():
    catalog = {"gpus": [{"id": BLACKWELL.id, "dataCenters": []}]}
    t = FakeTransport({"/catalog/gpus": catalog})
    with pytest.raises(RunpodError, match="no configured GPU"):
        Client("k", transport=t).pick_gpu([BLACKWELL], "EU-RO-1")


def test_create_pod_body_shape_for_gpu():
    t = FakeTransport({"/pods": {"id": "pod1"}})
    Client("k", transport=t).create_pod(
        name="comfy",
        template_id="wgd3p4n4o6",
        datacenter="EU-RO-1",
        volume_id="vol_abc",
        gpu_id=BLACKWELL.id,
    )
    method, path, body = t.calls[-1]
    assert (method, path) == ("POST", "/pods")
    assert body["gpu"] == {"id": BLACKWELL.id, "count": 1}
    assert body["templateId"] == "wgd3p4n4o6"
    assert body["cloud"] == "SECURE"
    assert body["dataCenterIds"] == ["EU-RO-1"]
    assert body["ports"] == ["22/tcp"]          # 8188 must never be exposed
    assert body["startSsh"] is True
    assert body["mounts"] == {"network": [{"volumeId": "vol_abc", "path": "/workspace"}]}
    assert "disk" not in body                   # let the template's value stand
    assert "cpu" not in body
    assert "terminateAfter" not in body         # no such field in v2


def test_create_pod_body_shape_for_cpu():
    t = FakeTransport({"/pods": {"id": "pod1"}})
    Client("k", transport=t).create_pod(
        name="provision",
        template_id=None,
        datacenter="EU-RO-1",
        volume_id="vol_abc",
        cpu={"id": "cpu5c", "vcpuCount": 2},
    )
    _, _, body = t.calls[-1]
    assert body["cpu"] == {"id": "cpu5c", "vcpuCount": 2}
    assert body["image"]                        # CPU pod needs an image, not a template
    assert "gpu" not in body
    assert "templateId" not in body


def test_create_pod_rejects_both_gpu_and_cpu():
    t = FakeTransport({})
    with pytest.raises(RunpodError, match="exactly one"):
        Client("k", transport=t).create_pod(
            name="x", template_id=None, datacenter="EU-RO-1", volume_id="v",
            gpu_id="g", cpu={"id": "cpu5c", "vcpuCount": 2},
        )


def test_create_pod_rejects_neither_gpu_nor_cpu():
    t = FakeTransport({})
    with pytest.raises(RunpodError, match="exactly one"):
        Client("k", transport=t).create_pod(
            name="x", template_id=None, datacenter="EU-RO-1", volume_id="v",
        )


def test_create_pod_omits_mounts_when_no_volume():
    """The harness must still work for a user who keeps no standing volume."""
    t = FakeTransport({"/pods": {"id": "pod1"}})
    Client("k", transport=t).create_pod(
        name="comfy", template_id="wgd3p4n4o6", datacenter="EU-RO-1",
        volume_id=None, gpu_id=BLACKWELL.id,
    )
    _, _, body = t.calls[-1]
    assert "mounts" not in body
    assert body["ports"] == ["22/tcp"]


def test_create_volume_body_shape():
    t = FakeTransport({"/network-volumes": {"id": "vol_new"}})
    vid = Client("k", transport=t).create_volume("comfy-models", 75, "EU-RO-1")
    assert vid == "vol_new"
    method, path, body = t.calls[-1]
    assert (method, path) == ("POST", "/network-volumes")
    assert body == {"name": "comfy-models", "size": 75, "dataCenter": "EU-RO-1"}


def test_terminate_uses_action_endpoint():
    t = FakeTransport({})
    Client("k", transport=t).terminate_pod("pod1")
    assert t.calls[-1] == ("POST", "/pods/pod1/action", {"action": "terminate"})


def test_ssh_target_reads_direct_tcp_mapping():
    pod = {
        "id": "pod1",
        "desiredStatus": "RUNNING",
        "runtime": {"ports": [{"private": 22, "public": 40022, "ip": "1.2.3.4", "type": "tcp"}]},
    }
    t = FakeTransport({"/pods/pod1": pod})
    host, port, user = Client("k", transport=t).ssh_target("pod1")
    assert (host, port, user) == ("1.2.3.4", 40022, "root")


def test_ssh_target_raises_while_runtime_is_null():
    t = FakeTransport({"/pods/pod1": {"id": "pod1", "runtime": None}})
    with pytest.raises(RunpodError, match="not ready"):
        Client("k", transport=t).ssh_target("pod1")


def test_ssh_target_raises_when_port_22_not_exposed():
    pod = {"runtime": {"ports": [{"private": 8188, "public": 1, "ip": "1.2.3.4", "type": "http"}]}}
    t = FakeTransport({"/pods/pod1": pod})
    with pytest.raises(RunpodError, match="22/tcp"):
        Client("k", transport=t).ssh_target("pod1")


def test_ssh_target_raises_when_port_22_has_no_public_mapping():
    pod = {"runtime": {"ports": [{"private": 22, "public": None, "ip": "1.2.3.4", "type": "tcp"}]}}
    t = FakeTransport({"/pods/pod1": pod})
    with pytest.raises(RunpodError, match="22/tcp"):
        Client("k", transport=t).ssh_target("pod1")


def test_pod_spend_reads_total_amount_from_metadata():
    billing = {
        "metadata": {
            "query": {"bucketSize": "day", "endTime": "...", "podId": "pod1", "startTime": "..."},
            "recordCount": 2,
            "totals": {"cpuAmount": 0.09, "diskAmount": 0, "gpuAmount": 0.21, "totalAmount": 0.30},
            "uniquePodCount": 1,
        },
        "records": [{"timestamp": "..."}, {"timestamp": "..."}],
    }
    t = FakeTransport({"/billing/pods": billing})
    assert Client("k", transport=t).pod_spend("pod1") == pytest.approx(0.30)


def test_pod_spend_returns_zero_when_no_charges():
    billing = {
        "metadata": {
            "query": {"bucketSize": "day", "endTime": "...", "podId": "pod1", "startTime": "..."},
            "recordCount": 0,
            "totals": {"cpuAmount": 0, "diskAmount": 0, "gpuAmount": 0, "totalAmount": 0},
            "uniquePodCount": 0,
        },
        "records": [],
    }
    t = FakeTransport({"/billing/pods": billing})
    assert Client("k", transport=t).pod_spend("pod1") == pytest.approx(0.0)


def test_pod_spend_returns_none_when_metadata_missing():
    t = FakeTransport({"/billing/pods": {}})
    assert Client("k", transport=t).pod_spend("pod1") is None
