from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    TASK1 = "task1"   # easy  — single-step trust decision
    TASK2 = "task2"   # medium — multi-step delegation chain (no adversary)
    TASK3 = "task3"   # hard  — full adversarial episode


class ActionType(str, Enum):
    DELEGATE           = "delegate"            # route subtask to a specialist
    VERIFY             = "verify"              # cross-check specialist result (+1 step cost)
    SOLVE_INDEPENDENTLY = "solve_independently" # agent solves itself (+2 step cost)
    SKIP               = "skip"               # abandon subtask (heavy penalty)


class SpecialistId(str, Enum):
    S0 = "S0"   # AccurateSlow
    S1 = "S1"   # OverconfidentFast
    S2 = "S2"   # DomainBound
    S3 = "S3"   # Adversarial (identity shuffled each episode)
    S4 = "S4"   # Degrading


class EpisodeStatus(str, Enum):
    ACTIVE    = "active"
    COMPLETED = "completed"
    FAILED    = "failed"


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SentinelObservation(BaseModel):
    session_id: str
    scenario_id: str
    task_type: TaskType
    difficulty: str                        # "easy" | "medium" | "hard"
    task_description: str
    current_subtask: str
    subtask_index: int                     # which subtask in the DAG (0-based)
    subtasks_total: int
    subtasks_remaining: int
    available_specialists: list[str]       # specialist ids visible to agent
    trust_snapshot: dict[str, float]       # {"S0": 0.82, "S1": 0.31, ...}
    stakes_level: float                    # 0.0–1.0; high = adversary may trigger
    step_count: int
    max_steps: int
    last_action_summary: str | None
    last_reward: float
    episode_status: EpisodeStatus


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SentinelAction(BaseModel):
    session_id: str
    task_type: TaskType
    action_type: ActionType
    specialist_id: str | None = None       # required for DELEGATE and VERIFY
    subtask_response: str | None = None    # required for SOLVE_INDEPENDENTLY
    reasoning: str | None = None           # optional chain-of-thought

    @field_validator("specialist_id")
    @classmethod
    def validate_specialist_id(cls, v: str | None) -> str | None:
        if v is not None and v not in [s.value for s in SpecialistId]:
            raise ValueError(f"specialist_id must be one of {[s.value for s in SpecialistId]}, got '{v}'")
        return v

    def requires_specialist(self) -> bool:
        return self.action_type in (ActionType.DELEGATE, ActionType.VERIFY)

    def requires_response(self) -> bool:
        return self.action_type == ActionType.SOLVE_INDEPENDENTLY


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class SentinelReward(BaseModel):
    value: float                          # (0.01, 0.99) boundary-exclusive
    reason: str
    signal_breakdown: dict[str, float]    # {"task_accuracy": 0.4, ...}

    @field_validator("value")
    @classmethod
    def clamp_reward(cls, v: float) -> float:
        return max(0.01, min(0.99, v))


# ---------------------------------------------------------------------------
# Step Result  (what env.step() and env.reset() return)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: SentinelObservation
    reward: SentinelReward
    done: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# State  (what env.state() returns)
# ---------------------------------------------------------------------------

class SentinelState(BaseModel):
    episode_id: str
    session_id: str | None
    step_count: int
    max_steps: int
    total_reward: float
    done: bool
    scenario_id: str
    task_type: TaskType
    difficulty: str
    status: EpisodeStatus
    last_reward: float
    subtasks_completed: int
    subtasks_total: int
    trust_snapshot: dict[str, float]
    adversarial_detections: int           # how many adversarial attempts caught
    adversarial_poisonings: int           # how many slipped through


# ---------------------------------------------------------------------------
# Reset Request
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_type: TaskType | None = None
    scenario_id: str | None = None
    seed: int | None = None