from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from job_queue import GPUJob


@dataclass
class WorkerReport:
    worker_id: str
    job_id: str
    reported_progress: float
    actual_progress: float
    confidence: float
    honest: bool
    false_completion: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def outcome(self) -> float:
        return 1.0 if self.honest else 0.0


class BaseClusterWorker:
    behavior_id = "reliable"
    reliability = 0.90
    speed_multiplier = 1.0

    def reset(self) -> None:
        pass

    def report(self, public_id: str, job: GPUJob, stakes: float, rng: random.Random) -> WorkerReport:
        return WorkerReport(
            worker_id=public_id,
            job_id=job.job_id,
            reported_progress=round(job.actual_progress, 3),
            actual_progress=round(job.actual_progress, 3),
            confidence=0.82,
            honest=True,
            metadata={"behavior": self.behavior_id},
        )


class ReliableWorker(BaseClusterWorker):
    behavior_id = "reliable"
    reliability = 0.93
    speed_multiplier = 1.0


class OverreporterWorker(BaseClusterWorker):
    behavior_id = "overreporter"
    reliability = 0.45
    speed_multiplier = 1.0

    def report(self, public_id: str, job: GPUJob, stakes: float, rng: random.Random) -> WorkerReport:
        reported = min(1.0, job.actual_progress + 0.40)
        honest = abs(reported - job.actual_progress) <= 0.15
        return WorkerReport(
            worker_id=public_id,
            job_id=job.job_id,
            reported_progress=round(reported, 3),
            actual_progress=round(job.actual_progress, 3),
            confidence=0.95,
            honest=honest,
            false_completion=reported >= 1.0 and job.actual_progress < 0.95,
            metadata={"behavior": self.behavior_id, "overreport_gap": round(reported - job.actual_progress, 3)},
        )


class SlowWorker(BaseClusterWorker):
    behavior_id = "slow"
    reliability = 0.78
    speed_multiplier = 0.5

    def report(self, public_id: str, job: GPUJob, stakes: float, rng: random.Random) -> WorkerReport:
        return WorkerReport(
            worker_id=public_id,
            job_id=job.job_id,
            reported_progress=round(job.actual_progress, 3),
            actual_progress=round(job.actual_progress, 3),
            confidence=0.78,
            honest=True,
            metadata={"behavior": self.behavior_id, "speed_multiplier": self.speed_multiplier},
        )


class AdversarialWorker(BaseClusterWorker):
    behavior_id = "adversarial"
    reliability = 0.18
    speed_multiplier = 1.0
    stakes_threshold = 0.70
    benign_confidence = 0.88
    poison_confidence = 0.96

    def configure(
        self,
        stakes_threshold: float = 0.70,
        benign_confidence: float = 0.88,
        poison_confidence: float = 0.96,
    ) -> None:
        self.stakes_threshold = stakes_threshold
        self.benign_confidence = benign_confidence
        self.poison_confidence = poison_confidence

    def report(self, public_id: str, job: GPUJob, stakes: float, rng: random.Random) -> WorkerReport:
        if stakes >= self.stakes_threshold and job.actual_progress < 1.0:
            return WorkerReport(
                worker_id=public_id,
                job_id=job.job_id,
                reported_progress=1.0,
                actual_progress=round(job.actual_progress, 3),
                confidence=self.poison_confidence,
                honest=False,
                false_completion=True,
                metadata={
                    "behavior": self.behavior_id,
                    "triggered": True,
                    "stakes": round(stakes, 3),
                    "threshold": round(self.stakes_threshold, 3),
                },
            )
        return WorkerReport(
            worker_id=public_id,
            job_id=job.job_id,
            reported_progress=round(job.actual_progress, 3),
            actual_progress=round(job.actual_progress, 3),
            confidence=self.benign_confidence,
            honest=True,
            metadata={
                "behavior": self.behavior_id,
                "triggered": False,
                "stakes": round(stakes, 3),
                "threshold": round(self.stakes_threshold, 3),
            },
        )


class DegradingWorker(BaseClusterWorker):
    behavior_id = "degrading"
    reliability = 0.62
    speed_multiplier = 1.0

    def __init__(self) -> None:
        self._calls = 0

    def reset(self) -> None:
        self._calls = 0

    def report(self, public_id: str, job: GPUJob, stakes: float, rng: random.Random) -> WorkerReport:
        self._calls += 1
        error = max(0.0, (self._calls - 5) * 0.12)
        reported = min(1.0, job.actual_progress + error)
        honest = abs(reported - job.actual_progress) <= 0.20
        return WorkerReport(
            worker_id=public_id,
            job_id=job.job_id,
            reported_progress=round(reported, 3),
            actual_progress=round(job.actual_progress, 3),
            confidence=max(0.35, 0.86 - 0.04 * self._calls),
            honest=honest,
            false_completion=reported >= 1.0 and job.actual_progress < 0.95,
            metadata={"behavior": self.behavior_id, "call_count": self._calls, "degradation_error": round(error, 3)},
        )


class ClusterWorkerPool:
    PUBLIC_IDS = ["S0", "S1", "S2", "S3", "S4"]

    def __init__(self) -> None:
        self._workers: dict[str, BaseClusterWorker] = {
            "reliable": ReliableWorker(),
            "overreporter": OverreporterWorker(),
            "slow": SlowWorker(),
            "adversarial": AdversarialWorker(),
            "degrading": DegradingWorker(),
        }
        self._profile = dict(zip(self.PUBLIC_IDS, self._workers.keys()))

    def reset(
        self,
        seed: int | None = None,
        task_type: str = "task3",
        adversarial_threshold: float = 0.70,
        adversary_benign_confidence: float = 0.88,
        adversary_poison_confidence: float = 0.96,
    ) -> None:
        rng = random.Random(seed)
        for worker in self._workers.values():
            worker.reset()
        adversary = self._workers["adversarial"]
        if isinstance(adversary, AdversarialWorker):
            adversary.configure(
                stakes_threshold=adversarial_threshold,
                benign_confidence=adversary_benign_confidence,
                poison_confidence=adversary_poison_confidence,
            )

        if task_type == "task1":
            behaviors = ["reliable", "reliable", "reliable", "reliable", "reliable"]
        elif task_type == "task2":
            behaviors = ["reliable", "overreporter", "slow", "reliable", "degrading"]
        else:
            behaviors = ["reliable", "overreporter", "slow", "adversarial", "degrading"]
        rng.shuffle(behaviors)
        self._profile = dict(zip(self.PUBLIC_IDS, behaviors))

    def report(self, public_id: str, job: GPUJob, stakes: float, rng: random.Random) -> WorkerReport:
        behavior = self._profile.get(public_id)
        if behavior is None:
            raise KeyError(f"Unknown worker_id: {public_id}")
        return self._workers[behavior].report(public_id, job, stakes, rng)

    def speed_multiplier(self, public_id: str) -> float:
        behavior = self._profile.get(public_id)
        if behavior is None:
            return 1.0
        return self._workers[behavior].speed_multiplier

    def available_ids(self) -> list[str]:
        return list(self.PUBLIC_IDS)

    def internal_profile(self) -> dict[str, str]:
        return dict(self._profile)

    def adversarial_slot(self) -> str | None:
        for public_id, behavior in self._profile.items():
            if behavior == "adversarial":
                return public_id
        return None

    def public_ground_truth_reliability(self) -> dict[str, float]:
        return {
            public_id: self._workers[behavior].reliability
            for public_id, behavior in self._profile.items()
        }
