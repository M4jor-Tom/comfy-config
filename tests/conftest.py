"""Shared test fixtures."""

import time

from comfy_runpod.runpod_api import RunpodError


class FakeRunpodClient:
    """Stands in for runpod_api.Client: same method names and keyword-only
    signatures as the real Client (checked directly in
    test_runpod_api.py's signature-mirroring test), no network. One fake
    shared by test_cli.py and test_provision.py so both exercise identical
    behaviour for the same call, instead of two hand-rolled copies quietly
    disagreeing on incidental details.
    """

    def __init__(
        self,
        *,
        pod_id: str = "pod-1",
        availability: str = "HIGH",
        desired_status: str = "RUNNING",
        spend: float | None = 0.42,
        pods: list[dict] | None = None,
        volumes: list[dict] | None = None,
    ) -> None:
        self.pod_id = pod_id
        self.availability = availability
        self.desired_status = desired_status
        self.spend = spend
        self.pods = pods if pods is not None else []
        self.volumes = volumes if volumes is not None else []
        self.calls: list[str] = []
        self.terminated: list[str] = []
        self.deleted_volumes: list[str] = []
        self.created_with: dict | None = None

    # --- catalog ---------------------------------------------------------

    def pick_gpu(self, gpus, datacenter):
        self.calls.append("pick_gpu")
        return gpus[0], self.availability

    # --- volumes ---------------------------------------------------------

    def create_volume(self, name, size_gb, datacenter):
        self.calls.append("create_volume")
        return "vol-created"

    def list_volumes(self):
        self.calls.append("list_volumes")
        return self.volumes

    def delete_volume(self, volume_id):
        self.calls.append("delete_volume")
        self.deleted_volumes.append(volume_id)

    # --- pods --------------------------------------------------------------

    def list_pods(self):
        self.calls.append("list_pods")
        return self.pods

    def create_pod(self, *, name, template_id, datacenter, volume_id, gpu_id=None, cpu=None):
        # Keyword-only with the same names as the real Client.create_pod, on
        # purpose: a misspelled/renamed kwarg at the call site must raise
        # TypeError here too, not just against the real client.
        self.calls.append("create_pod")
        self.created_with = {
            "name": name, "template_id": template_id, "datacenter": datacenter,
            "volume_id": volume_id, "gpu_id": gpu_id, "cpu": cpu,
        }
        return {"id": self.pod_id}

    def get_pod(self, pod_id):
        self.calls.append("get_pod")
        return {"id": pod_id, "desiredStatus": self.desired_status}

    def terminate_pod(self, pod_id):
        self.calls.append("terminate_pod")
        self.terminated.append(pod_id)

    def ssh_target(self, pod_id):
        self.calls.append("ssh_target")
        return ("1.2.3.4", 40022, "root")

    def wait_for_ssh(self, pod_id, timeout=900):
        self.calls.append("wait_for_ssh")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                return self.ssh_target(pod_id)
            except RunpodError:
                time.sleep(10)
        raise RunpodError(f"pod {pod_id} never exposed SSH within {timeout}s")

    def pod_spend(self, pod_id):
        self.calls.append("pod_spend")
        return self.spend
