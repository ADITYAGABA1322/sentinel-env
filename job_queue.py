from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class GPUJob:
    job_id: str
    priority: int
    memory_required: int
    steps_to_complete: int
    deadline: int
    owner: str
    status: JobStatus = JobStatus.QUEUED
    assigned_gpu: str | None = None
    actual_progress: float = 0.0
    reported_progress: float = 0.0
    completed_at: int | None = None


class JobQueue:
    """Job queue with hidden priorities, deadlines, and progress tracking."""

    def __init__(self, jobs: list[GPUJob] | None = None) -> None:
        self._jobs: dict[str, GPUJob] = {}
        for job in jobs or []:
            self.submit(job)

    @classmethod
    def generate(
        cls,
        count: int,
        seed: int | None = None,
        min_memory: int = 10,
        max_memory: int = 75,
        min_steps: int = 2,
        max_steps: int = 12,
        deadline_min: int = 12,
        deadline_max: int = 120,
    ) -> "JobQueue":
        if count <= 0:
            raise ValueError("count must be positive.")
        rng = random.Random(seed)
        jobs = [
            GPUJob(
                job_id=f"JOB-{idx:03d}",
                priority=rng.randint(1, 5),
                memory_required=rng.randint(min_memory, max_memory),
                steps_to_complete=rng.randint(min_steps, max_steps),
                deadline=rng.randint(deadline_min, deadline_max),
                owner=f"team-{rng.randint(1, 4)}",
            )
            for idx in range(count)
        ]
        return cls(jobs)

    def submit(self, job: GPUJob) -> str:
        if job.job_id in self._jobs:
            raise ValueError(f"Duplicate job_id: {job.job_id}")
        if not 1 <= job.priority <= 5:
            raise ValueError("priority must be in range 1..5.")
        if job.memory_required <= 0:
            raise ValueError("memory_required must be positive.")
        if job.steps_to_complete <= 0:
            raise ValueError("steps_to_complete must be positive.")
        self._jobs[job.job_id] = job
        return job.job_id

    def get(self, job_id: str) -> GPUJob:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job_id: {job_id}")
        return self._jobs[job_id]

    def assign(self, job_id: str, gpu_id: str) -> bool:
        job = self.get(job_id)
        if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return False
        job.status = JobStatus.RUNNING
        job.assigned_gpu = gpu_id
        return True

    def unassign(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job.status != JobStatus.RUNNING:
            return False
        job.status = JobStatus.QUEUED
        job.assigned_gpu = None
        return True

    def tick(self, current_step: int, active_job_ids: set[str] | None = None) -> list[str]:
        """
        Advance job progress and mark deadlines.

        active_job_ids lets the environment pass jobs currently allocated on
        GPUs. If omitted, all RUNNING jobs advance.
        """
        timed_out: list[str] = []
        for job in self._jobs.values():
            if job.status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.TIMED_OUT):
                continue
            if current_step > job.deadline:
                job.status = JobStatus.TIMED_OUT
                job.assigned_gpu = None
                timed_out.append(job.job_id)
                continue
            if job.status == JobStatus.RUNNING and (
                active_job_ids is None or job.job_id in active_job_ids
            ):
                increment = 1.0 / job.steps_to_complete
                job.actual_progress = min(1.0, job.actual_progress + increment)
                job.reported_progress = max(job.reported_progress, job.actual_progress)
                if job.actual_progress >= 1.0:
                    job.status = JobStatus.COMPLETE
                    job.completed_at = current_step
                    job.assigned_gpu = None
        return timed_out

    def advance(
        self,
        job_id: str,
        current_step: int,
        progress_multiplier: float = 1.0,
    ) -> bool:
        """
        Advance one running job by a worker-specific speed multiplier.

        Returns True when the job is complete after this advancement.
        """
        job = self.get(job_id)
        if job.status != JobStatus.RUNNING:
            return job.status == JobStatus.COMPLETE
        if current_step > job.deadline:
            job.status = JobStatus.TIMED_OUT
            job.assigned_gpu = None
            return False

        increment = max(0.0, progress_multiplier) / job.steps_to_complete
        job.actual_progress = min(1.0, job.actual_progress + increment)
        job.reported_progress = max(job.reported_progress, job.actual_progress)
        if job.actual_progress >= 1.0:
            job.status = JobStatus.COMPLETE
            job.completed_at = current_step
            job.assigned_gpu = None
            return True
        return False

    def complete(self, job_id: str, actual: bool = True, current_step: int | None = None) -> float:
        job = self.get(job_id)
        if actual:
            job.actual_progress = 1.0
            job.reported_progress = 1.0
            job.status = JobStatus.COMPLETE
            job.completed_at = current_step
            job.assigned_gpu = None
            return 1.0
        job.reported_progress = 1.0
        return 0.0

    def fail(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job.status in (JobStatus.COMPLETE, JobStatus.TIMED_OUT):
            return False
        job.status = JobStatus.FAILED
        job.assigned_gpu = None
        return True

    def pending_jobs(self) -> list[GPUJob]:
        return [job for job in self._jobs.values() if job.status == JobStatus.QUEUED]

    def running_jobs(self) -> list[GPUJob]:
        return [job for job in self._jobs.values() if job.status == JobStatus.RUNNING]

    def active_job_ids(self) -> set[str]:
        return {job.job_id for job in self.running_jobs()}

    def deadline_pressure(self, current_step: int, window: int = 10) -> list[GPUJob]:
        return [
            job for job in self._jobs.values()
            if job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            and current_step <= job.deadline <= current_step + window
        ]

    def completion_rate(self) -> float:
        if not self._jobs:
            return 0.0
        completed = sum(1 for job in self._jobs.values() if job.status == JobStatus.COMPLETE)
        return completed / len(self._jobs)

    def deadline_hit_rate(self) -> float:
        completed = [job for job in self._jobs.values() if job.status == JobStatus.COMPLETE]
        if not completed:
            return 0.0
        hits = sum(1 for job in completed if job.completed_at is not None and job.completed_at <= job.deadline)
        return hits / len(completed)

    def snapshot(self, include_hidden: bool = False) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for job in self._jobs.values():
            row = {
                "job_id": job.job_id,
                "memory_required": job.memory_required,
                "steps_to_complete": job.steps_to_complete,
                "deadline": job.deadline,
                "owner": job.owner,
                "status": job.status.value,
                "assigned_gpu": job.assigned_gpu,
                "reported_progress": round(job.reported_progress, 3),
            }
            if include_hidden:
                row["priority"] = job.priority
                row["actual_progress"] = round(job.actual_progress, 3)
            rows.append(row)
        return rows

    def summary(self) -> dict[str, Any]:
        statuses = {status.value: 0 for status in JobStatus}
        for job in self._jobs.values():
            statuses[job.status.value] += 1
        return {
            "jobs_total": len(self._jobs),
            "statuses": statuses,
            "completion_rate": round(self.completion_rate(), 4),
            "deadline_hit_rate": round(self.deadline_hit_rate(), 4),
        }
