"""RunPod API wrapper with live and dry-run (simulated) modes."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

_GPU_TYPE_ALIASES = {
    "A40": "NVIDIA A40",
}


def _normalize_gpu_type(gpu_type: str) -> str:
    g = (gpu_type or "").strip()
    return _GPU_TYPE_ALIASES.get(g, g or "NVIDIA RTX A4000")


class RunPodClient:
    """Manages RunPod pods.  In dry-run mode all state is simulated in-memory."""

    def __init__(self, api_key: str, dry_run: bool = True):
        self.api_key = api_key
        self.dry_run = dry_run
        self._lock = threading.Lock()
        # Simulated state (dry-run only)
        self._pods: dict[str, dict[str, Any]] = {}
        # Track pods created by the agent vs system
        self._agent_created_pod_ids: set[str] = set()
        # Cost tracking
        self._estimated_spend_usd: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_pods(self) -> list[dict[str, Any]]:
        if self.dry_run:
            with self._lock:
                return list(self._pods.values())
        pods = self._live_list_pods()
        for p in pods:
            pid = p.get("id", "")
            p["created_by"] = "agent" if pid in self._agent_created_pod_ids else "system"
        return pods

    def create_pod(
        self,
        name: str,
        image: str,
        gpu_type: str = "NVIDIA RTX A4000",
        gpu_count: int = 1,
        volume_in_gb: int = 20,
        ports: str = "8000/http",
    ) -> dict[str, Any]:
        gpu_type = _normalize_gpu_type(gpu_type)
        if self.dry_run:
            return self._sim_create_pod(
                name, image, gpu_type, gpu_count, volume_in_gb, ports
            )
        return self._live_create_pod(
            name, image, gpu_type, gpu_count, volume_in_gb, ports
        )

    def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        if self.dry_run:
            with self._lock:
                return self._pods.get(pod_id)
        pod = self._live_get_pod(pod_id)
        if pod is None:
            return None
        pod["created_by"] = "agent" if pod.get("id", pod_id) in self._agent_created_pod_ids else "system"
        return pod

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        if self.dry_run:
            return self._sim_stop_pod(pod_id)
        return self._live_stop_pod(pod_id)

    def terminate_pod(self, pod_id: str) -> dict[str, Any]:
        if self.dry_run:
            return self._sim_terminate_pod(pod_id)
        return self._live_terminate_pod(pod_id)

    @property
    def estimated_spend(self) -> float:
        return self._estimated_spend_usd

    # ------------------------------------------------------------------
    # Simulated (dry-run) implementations
    # ------------------------------------------------------------------

    def seed_pod(self, pod_id: str, name: str = "agent-service") -> None:
        """Seed the simulation with an existing pod (the agent's own pod)."""
        with self._lock:
            self._pods[pod_id] = {
                "id": pod_id,
                "name": name,
                "desiredStatus": "RUNNING",
                "runtime": {"status": "RUNNING"},
                "gpu": "NVIDIA RTX A4000",
                "gpuCount": 1,
                "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                "ports": "8000/http",
                "volumeInGb": 20,
                "created_by": "system",
            }

    def _sim_create_pod(
        self, name: str, image: str, gpu_type: str,
        gpu_count: int, volume_in_gb: int, ports: str,
    ) -> dict[str, Any]:
        pod_id = f"pod-{uuid.uuid4().hex[:8]}"
        pod = {
            "id": pod_id,
            "name": name,
            "desiredStatus": "RUNNING",
            "runtime": {"status": "RUNNING"},
            "gpu": gpu_type,
            "gpuCount": gpu_count,
            "imageName": image,
            "ports": ports,
            "volumeInGb": volume_in_gb,
            "created_by": "agent",
        }
        with self._lock:
            self._pods[pod_id] = pod
            self._agent_created_pod_ids.add(pod_id)
        # Rough cost estimate: ~$0.50/hr per GPU
        self._estimated_spend_usd += 0.50 * gpu_count
        return {"status": "ok", "pod": pod}

    def _sim_stop_pod(self, pod_id: str) -> dict[str, Any]:
        with self._lock:
            pod = self._pods.get(pod_id)
            if not pod:
                return {"status": "error", "message": f"Pod {pod_id} not found"}
            pod["desiredStatus"] = "STOPPED"
            pod["runtime"] = {"status": "STOPPED"}
            return {"status": "ok", "pod": pod}

    def _sim_terminate_pod(self, pod_id: str) -> dict[str, Any]:
        with self._lock:
            pod = self._pods.pop(pod_id, None)
            if not pod:
                return {"status": "error", "message": f"Pod {pod_id} not found"}
            self._agent_created_pod_ids.discard(pod_id)
            return {"status": "ok", "message": f"Pod {pod_id} terminated"}

    # ------------------------------------------------------------------
    # Live implementations (RunPod SDK)
    # ------------------------------------------------------------------

    def _live_list_pods(self) -> list[dict[str, Any]]:
        import runpod
        runpod.api_key = self.api_key
        return runpod.get_pods()

    def _live_create_pod(
        self, name: str, image: str, gpu_type: str,
        gpu_count: int, volume_in_gb: int, ports: str,
    ) -> dict[str, Any]:
        import runpod
        runpod.api_key = self.api_key
        requested_gpu_count = max(1, int(gpu_count or 1))
        gpu_candidates = []
        for candidate in [gpu_type, "NVIDIA RTX A4000", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A40"]:
            normalized = _normalize_gpu_type(candidate)
            if normalized not in gpu_candidates:
                gpu_candidates.append(normalized)
        errors: list[str] = []
        pod = None
        chosen_gpu_type = gpu_candidates[0]
        try:
            for candidate in gpu_candidates:
                chosen_gpu_type = candidate
                try:
                    pod = runpod.create_pod(
                        name=name,
                        image_name=image,
                        gpu_type_id=candidate,
                        gpu_count=requested_gpu_count,
                        volume_in_gb=volume_in_gb,
                        ports=ports,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{candidate}: {exc}")
            if pod is None:
                return {"status": "error", "error": " | ".join(errors)}
        except Exception as exc:  # noqa: BLE001 - provider errors are benchmark observations.
            return {"status": "error", "error": str(exc)}
        with self._lock:
            self._agent_created_pod_ids.add(pod["id"])
        # Rough cost estimate: ~$0.50/hr per GPU (mirrors dry-run behavior; used only for safety caps).
        self._estimated_spend_usd += 0.50 * requested_gpu_count
        pod["created_by"] = "agent"
        pod["requested_gpu_type"] = gpu_type
        pod["actual_gpu_type"] = chosen_gpu_type
        pod["requested_gpu_count"] = gpu_count
        pod["actual_gpu_count"] = requested_gpu_count
        return {"status": "ok", "pod": pod}

    def _live_get_pod(self, pod_id: str) -> dict[str, Any] | None:
        import runpod
        runpod.api_key = self.api_key
        try:
            return runpod.get_pod(pod_id)
        except Exception:
            return None

    def _live_stop_pod(self, pod_id: str) -> dict[str, Any]:
        import runpod
        runpod.api_key = self.api_key
        try:
            runpod.stop_pod(pod_id)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc), "pod_id": pod_id}
        return {"status": "ok", "message": f"Pod {pod_id} stopped"}

    def _live_terminate_pod(self, pod_id: str) -> dict[str, Any]:
        import runpod
        runpod.api_key = self.api_key
        try:
            runpod.terminate_pod(pod_id)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc), "pod_id": pod_id}
        with self._lock:
            self._agent_created_pod_ids.discard(pod_id)
        return {"status": "ok", "message": f"Pod {pod_id} terminated"}
