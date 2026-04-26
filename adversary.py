from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any


ATTACK_LEVELS = [
    "false_completion",
    "false_memory_report",
    "cascade_panic",
    "fabricate_job",
    "coordinated_multi_report",
]


@dataclass
class AttackEvent:
    attack_id: str
    level: int
    attack_type: str
    step: int
    payload: dict[str, Any]
    detected: bool = False


class AdversaryFSM:
    """
    Self-play adversary used by the GPU-cluster version of SENTINEL.

    The FSM escalates when the Auditor catches attacks, creating a curriculum
    without needing a learned adversary in Phase 1.
    """

    def __init__(
        self,
        seed: int | None = None,
        attack_probability: float = 0.25,
        min_attack_gap: int = 5,
    ) -> None:
        self._rng = random.Random(seed)
        self._attack_probability = attack_probability
        self._min_attack_gap = min_attack_gap
        self._level = 1
        self._last_attack_step = -min_attack_gap
        self._history: list[AttackEvent] = []

    def current_level(self) -> int:
        return self._level

    def current_attack_type(self) -> str:
        return ATTACK_LEVELS[self._level - 1]

    def maybe_inject(
        self,
        step: int,
        auditor_detection_rate: float = 0.0,
        gpu_ids: list[str] | None = None,
        job_ids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        if auditor_detection_rate > 0.70:
            self.escalate()
        if step - self._last_attack_step < self._min_attack_gap:
            return None
        if self._rng.random() > self._attack_probability:
            return None

        event = AttackEvent(
            attack_id=str(uuid.uuid4()),
            level=self._level,
            attack_type=self.current_attack_type(),
            step=step,
            payload=self._payload_for_current_level(gpu_ids or [], job_ids or []),
        )
        self._last_attack_step = step
        self._history.append(event)
        return event.__dict__

    def record_detection(self, attack_id: str, detected: bool) -> None:
        for event in self._history:
            if event.attack_id == attack_id:
                event.detected = detected
                if detected:
                    self.escalate()
                return
        raise KeyError(f"Unknown attack_id: {attack_id}")

    def escalate(self) -> None:
        self._level = min(len(ATTACK_LEVELS), self._level + 1)

    def detection_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for event in self._history if event.detected) / len(self._history)

    def history(self) -> list[dict[str, Any]]:
        return [event.__dict__ for event in self._history]

    def _payload_for_current_level(self, gpu_ids: list[str], job_ids: list[str]) -> dict[str, Any]:
        gpu_id = self._rng.choice(gpu_ids) if gpu_ids else "GPU-00"
        job_id = self._rng.choice(job_ids) if job_ids else "JOB-000"
        attack_type = self.current_attack_type()

        if attack_type == "false_completion":
            return {
                "target": job_id,
                "reported_status": "complete",
                "reported_progress": 1.0,
            }
        if attack_type == "false_memory_report":
            return {
                "target": gpu_id,
                "reported_state": "IDLE",
                "reported_memory_free": 40,
            }
        if attack_type == "cascade_panic":
            return {
                "reported_cluster_health": 0.0,
                "message": "cascade_failure_imminent",
            }
        if attack_type == "fabricate_job":
            return {
                "job_id": f"FAB-{self._rng.randint(1000, 9999)}",
                "priority": 5,
                "memory_required": 72,
                "deadline": 8,
                "owner": "forged-critical-team",
            }
        return {
            "sequence": [
                {"target": gpu_id, "reported_state": "IDLE", "reported_memory_free": 40},
                {"target": job_id, "reported_status": "complete", "reported_progress": 1.0},
                {"reported_cluster_health": 0.25},
            ]
        }
