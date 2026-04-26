from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GPUState(str, Enum):
    IDLE = "IDLE"
    ALLOCATED = "ALLOCATED"
    OVERLOADED = "OVERLOADED"
    FAILED = "FAILED"
    RECOVERING = "RECOVERING"


@dataclass
class GPUDevice:
    gpu_id: str
    memory_total: int = 80
    state: GPUState = GPUState.IDLE
    jobs_running: dict[str, int] = field(default_factory=dict)
    failure_probability: float = 0.0
    recovery_steps_remaining: int = 0
    false_report: dict[str, Any] | None = None

    @property
    def memory_used(self) -> int:
        return sum(self.jobs_running.values())

    @property
    def memory_free(self) -> int:
        return max(0, self.memory_total - self.memory_used)

    def is_operational(self) -> bool:
        return self.state not in (GPUState.FAILED, GPUState.RECOVERING)


class GPUPool:
    """
    Stateful GPU cluster simulator.

    Phase 1 intentionally keeps this independent from SentinelEnv so we can
    test the cluster mechanics before wiring them into the OpenEnv API.
    """

    def __init__(
        self,
        num_gpus: int = 16,
        memory_per_gpu: int = 80,
        failure_probability: float = 0.0,
        recovery_steps: int = 3,
    ) -> None:
        if num_gpus <= 0:
            raise ValueError("num_gpus must be positive.")
        if memory_per_gpu <= 0:
            raise ValueError("memory_per_gpu must be positive.")

        self._recovery_steps = recovery_steps
        self._gpus: dict[str, GPUDevice] = {
            f"GPU-{idx:02d}": GPUDevice(
                gpu_id=f"GPU-{idx:02d}",
                memory_total=memory_per_gpu,
                failure_probability=failure_probability,
            )
            for idx in range(num_gpus)
        }

    def allocate(
        self,
        job_id: str,
        gpu_id: str,
        memory_required: int,
        allow_overcommit: bool = True,
    ) -> bool:
        if memory_required <= 0:
            raise ValueError("memory_required must be positive.")
        gpu = self._require_gpu(gpu_id)
        if not gpu.is_operational():
            return False
        if self.find_job_gpu(job_id) is not None:
            return False
        if not allow_overcommit and memory_required > gpu.memory_free:
            return False

        gpu.jobs_running[job_id] = memory_required
        self._refresh_state(gpu)
        return True

    def preempt(self, job_id: str) -> bool:
        gpu_id = self.find_job_gpu(job_id)
        if gpu_id is None:
            return False
        gpu = self._gpus[gpu_id]
        gpu.jobs_running.pop(job_id, None)
        self._refresh_state(gpu)
        return True

    def find_job_gpu(self, job_id: str) -> str | None:
        for gpu_id, gpu in self._gpus.items():
            if job_id in gpu.jobs_running:
                return gpu_id
        return None

    def tick(self, rng: random.Random | None = None) -> list[str]:
        """
        Advance hardware state by one step.

        Returns GPU ids that newly failed on this tick.
        """
        rng = rng or random.Random()
        newly_failed: list[str] = []
        for gpu in self._gpus.values():
            if gpu.state == GPUState.FAILED:
                gpu.state = GPUState.RECOVERING
                gpu.recovery_steps_remaining = self._recovery_steps
                continue

            if gpu.state == GPUState.RECOVERING:
                gpu.recovery_steps_remaining -= 1
                if gpu.recovery_steps_remaining <= 0:
                    gpu.jobs_running.clear()
                    gpu.state = GPUState.IDLE
                continue

            if gpu.jobs_running and rng.random() < gpu.failure_probability:
                gpu.state = GPUState.FAILED
                newly_failed.append(gpu.gpu_id)
                continue

            self._refresh_state(gpu)
        return newly_failed

    def inject_false_report(self, gpu_id: str, false_state: dict[str, Any]) -> None:
        gpu = self._require_gpu(gpu_id)
        gpu.false_report = dict(false_state)

    def clear_false_reports(self) -> None:
        for gpu in self._gpus.values():
            gpu.false_report = None

    def utilization_rate(self) -> float:
        total_memory = sum(gpu.memory_total for gpu in self._gpus.values() if gpu.is_operational())
        if total_memory <= 0:
            return 0.0
        used = sum(min(gpu.memory_used, gpu.memory_total) for gpu in self._gpus.values() if gpu.is_operational())
        return round(used / total_memory, 4)

    def cluster_health_score(self) -> float:
        total = len(self._gpus)
        failed_like = sum(
            1 for gpu in self._gpus.values()
            if gpu.state in (GPUState.FAILED, GPUState.RECOVERING)
        )
        idle_or_failed = sum(
            1 for gpu in self._gpus.values()
            if gpu.state in (GPUState.IDLE, GPUState.FAILED, GPUState.RECOVERING)
        )
        overloaded = sum(1 for gpu in self._gpus.values() if gpu.state == GPUState.OVERLOADED)

        if failed_like / total > 0.60:
            return 0.0
        if idle_or_failed / total > 0.30 or overloaded / total > 0.25:
            return 0.5
        return 1.0

    def snapshot(self, include_hidden: bool = False) -> list[dict[str, Any]]:
        return [self._gpu_snapshot(gpu, include_hidden=include_hidden) for gpu in self._gpus.values()]

    def summary(self) -> dict[str, Any]:
        states = {state.value: 0 for state in GPUState}
        for gpu in self._gpus.values():
            states[gpu.state.value] += 1
        return {
            "num_gpus": len(self._gpus),
            "states": states,
            "utilization_rate": self.utilization_rate(),
            "cluster_health_score": self.cluster_health_score(),
            "memory_used": sum(gpu.memory_used for gpu in self._gpus.values()),
            "memory_total": sum(gpu.memory_total for gpu in self._gpus.values()),
        }

    def _require_gpu(self, gpu_id: str) -> GPUDevice:
        if gpu_id not in self._gpus:
            raise KeyError(f"Unknown gpu_id: {gpu_id}")
        return self._gpus[gpu_id]

    def _refresh_state(self, gpu: GPUDevice) -> None:
        if gpu.state in (GPUState.FAILED, GPUState.RECOVERING):
            return
        if not gpu.jobs_running:
            gpu.state = GPUState.IDLE
        elif gpu.memory_used > gpu.memory_total:
            gpu.state = GPUState.OVERLOADED
        else:
            gpu.state = GPUState.ALLOCATED

    def _gpu_snapshot(self, gpu: GPUDevice, include_hidden: bool) -> dict[str, Any]:
        actual = {
            "id": gpu.gpu_id,
            "state": gpu.state.value,
            "memory_total": gpu.memory_total,
            "memory_used": gpu.memory_used,
            "memory_free": gpu.memory_free,
            "jobs": list(gpu.jobs_running.keys()),
        }
        if include_hidden:
            actual["false_report"] = gpu.false_report
            actual["recovery_steps_remaining"] = gpu.recovery_steps_remaining
            return actual
        if gpu.false_report:
            visible = dict(actual)
            visible.update(gpu.false_report)
            visible["report_tampered"] = True
            return visible
        return actual
