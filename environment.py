from __future__ import annotations

import copy
import random
import re
import uuid
from typing import Any

from difficulty_controller import (
    GLOBAL_DIFFICULTY_CONTROLLER,
    DifficultyController,
    DifficultyProfile,
)
from graders import (
    grade_task1_step,
    grade_task2_step, grade_task2_terminal,
    grade_task3_step, grade_task3_terminal,
)
from scenarios import sample_scenario, get_scenario, Scenario
from sentinel_config import CRITICAL_POISON_STAKES, VERIFY_EXTRA_STEP_COST
from specialists import SpecialistPool
from task_graph import TaskGraph
from trust_ledger import TrustLedger

# ---------------------------------------------------------------------------
# Lazy Pydantic imports — falls back to plain dicts if not installed
# ---------------------------------------------------------------------------
try:
    from models import (
        SentinelObservation, SentinelAction, SentinelReward,
        SentinelState, StepResult, EpisodeStatus,
        TaskType, ActionType, SpecialistId, ResetRequest,
    )
    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False


# Hidden ground-truth reliability for each specialist FSM type.
# These are NEVER sent to the agent — only used by the reward engine
# to compute calibration score.
_GROUND_TRUTH_RELIABILITY: dict[str, float] = {
    "S0": 0.90,   # AccurateSlow
    "S1": 0.60,   # OverconfidentFast
    "S2": 0.70,   # DomainBound (avg of in/out domain)
    "S3": 0.15,   # Adversarial (unreliable overall)
    "S4": 0.65,   # Degrading (avg over calls)
}

MAX_STEPS = {
    "task1": 15,
    "task2": 30,
    "task3": 45,
}


class SentinelEnv:
    """
    Core environment class — mirrors Round 1 IncidentEnv pattern exactly.
    reset() / step() / state() API.
    No BaseEnv subclassing needed — plain Python class, FastAPI wraps it.
    """

    def __init__(self) -> None:
        self.current_scenario: Scenario | None = None
        self.episode_id: str = ""
        self.session_id: str = ""
        self.step_count: int = 0
        self.max_steps: int = 0
        self.total_reward: float = 0.0
        self.reward_events: int = 0
        self.last_reward: float = 0.0
        self.done: bool = False
        self.episode_status: str = "active"
        self.last_action_summary: str | None = None
        self._reward_trace: list[dict[str, Any]] = []

        self._graph: TaskGraph | None = None
        self._ledger: TrustLedger = TrustLedger()
        self._pool: SpecialistPool = SpecialistPool()
        self._rng: random.Random = random.Random()
        self._difficulty_controller: DifficultyController = GLOBAL_DIFFICULTY_CONTROLLER
        self._difficulty_profile: DifficultyProfile = DifficultyProfile()

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        task_type: str | None = None,
        scenario_id: str | None = None,
        seed: int | None = None,
        adaptive: bool = False,
    ) -> dict:

        self._rng = random.Random(seed)

        # Select scenario
        if scenario_id:
            scenario = get_scenario(scenario_id)
        else:
            task = task_type or "task3"
            scenario = sample_scenario(task, seed=seed)

        self._difficulty_profile = self._difficulty_controller.profile(adaptive=adaptive)
        scenario = self._apply_difficulty_profile(scenario, self._difficulty_profile)

        self.current_scenario = scenario
        self.episode_id       = str(uuid.uuid4())
        self.session_id       = str(uuid.uuid4())
        self.step_count       = 0
        self.max_steps        = max(
            len(scenario["subtasks"]),
            MAX_STEPS[scenario["task_type"]] - self._difficulty_profile.verify_budget_penalty,
        )
        self.total_reward     = 0.0
        self.reward_events    = 0
        self.last_reward      = 0.0
        self.done             = False
        self.episode_status   = "active"
        self.last_action_summary = None
        self._reward_trace = []

        # Reset subcomponents
        self._graph = TaskGraph(scenario)
        self._ledger.reset()
        self._pool.reset(seed=seed)
        self._pool.configure_adversary(
            stakes_threshold=self._difficulty_profile.adversarial_threshold,
            benign_confidence=self._difficulty_profile.adversary_benign_confidence,
            poison_confidence=self._difficulty_profile.adversary_poison_confidence,
        )

        return self._build_step_result(
            reward_value=0.0,
            reason="Episode initialized.",
            breakdown={},
            done=False,
            extra_info={"episode_id": self.episode_id, "session_id": self.session_id},
        )

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: dict) -> dict:
        if self.current_scenario is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode already completed. Call reset().")

        # --- Validate session ---
        if action.get("session_id") and action["session_id"] != self.session_id:
            raise ValueError(
                f"session_id mismatch: expected '{self.session_id}', got '{action['session_id']}'"
            )

        action_type     = action.get("action_type", "delegate")
        specialist_id   = action.get("specialist_id")
        task_type       = self.current_scenario["task_type"]

        # --- Validate action fields ---
        if action_type in ("delegate", "verify") and not specialist_id:
            raise ValueError(f"action_type='{action_type}' requires specialist_id.")
        if action_type == "solve_independently" and not action.get("subtask_response"):
            raise ValueError("action_type='solve_independently' requires subtask_response.")

        # --- Get current subtask ---
        node = self._graph.current_node()
        if node is None:
            # All nodes done — emit terminal reward
            return self._terminal_reward()

        subtask  = node.subtask
        stakes   = subtask["stakes"]
        confidence: float | None = None
        result_metadata: dict[str, Any] = {}
        trust_before = self._ledger.trust(specialist_id) if specialist_id else None

        step_cost = 1

        # --- Execute specialist or self-solve ---
        if action_type == "skip":
            self._graph.skip_node(subtask["id"])
            outcome        = 0.0
            was_adversarial = False
            self.last_action_summary = f"Skipped {subtask['id']}"

        elif action_type == "solve_independently":
            # Agent solves itself — always correct (no specialist involved)
            # But costs 2 steps (enforced via max_steps budget pressure).
            step_cost       = 2
            outcome         = 1.0
            was_adversarial = False
            self._graph.record_outcome(subtask["id"], outcome, "SELF")
            self.last_action_summary = f"Self-solved {subtask['id']}"

        elif action_type == "verify":
            # First get result, then verify (costs +1 step)
            result = self._pool.execute(
                specialist_id,
                subtask["description"],
                stakes,
                self._rng,
                domain=subtask.get("domain"),
            )
            step_cost       = int(result.metadata.get("step_cost", 1)) + VERIFY_EXTRA_STEP_COST
            outcome         = result.outcome if not result.is_adversarial else 0.0
            was_adversarial  = result.is_adversarial
            confidence = result.confidence
            result_metadata = dict(result.metadata)
            # Verification means agent caught adversarial — treat as detection
            if result.is_adversarial:
                outcome = 1.0   # successfully avoided
            self._graph.record_outcome(subtask["id"], outcome, specialist_id, was_adversarial)
            # Important: trust must learn from the specialist's raw answer, not
            # from the corrected/avoided graph outcome. If S0 was caught lying,
            # the task node is safe, but S0's trust should still drop.
            self._ledger.update(
                specialist_id,
                result.outcome,
                stakes,
                confidence=result.confidence,
                domain=subtask.get("domain"),
            )
            self.last_action_summary = f"Verified {specialist_id} on {subtask['id']}"

        else:  # delegate
            result          = self._pool.execute(
                specialist_id,
                subtask["description"],
                stakes,
                self._rng,
                domain=subtask.get("domain"),
            )
            step_cost       = int(result.metadata.get("step_cost", 1))
            was_adversarial  = result.is_adversarial
            outcome         = 0.0 if was_adversarial else result.outcome
            confidence = result.confidence
            result_metadata = dict(result.metadata)
            self._graph.record_outcome(subtask["id"], outcome, specialist_id, was_adversarial)
            self._ledger.update(
                specialist_id,
                result.outcome,
                stakes,
                confidence=result.confidence,
                domain=subtask.get("domain"),
            )
            self.last_action_summary = f"Delegated to {specialist_id} on {subtask['id']}"

        self.step_count += max(1, step_cost)

        # --- Grade this step ---
        reward_value, reason, breakdown = self._grade_step(
            task_type, action_type, specialist_id, outcome,
            stakes, was_adversarial, confidence, result_metadata, trust_before,
        )

        self.last_reward   = reward_value
        self.total_reward += reward_value
        self.reward_events += 1
        self._record_reward_event(
            kind="step",
            action_type=action_type,
            specialist_id=specialist_id,
            subtask=subtask,
            stakes=stakes,
            reward_value=reward_value,
            reason=reason,
            breakdown=breakdown,
            was_adversarial=was_adversarial,
            confidence=confidence,
            result_metadata=result_metadata,
            trust_before=trust_before,
        )

        # --- Check episode end ---
        all_done    = self._graph.is_done()
        budget_gone = self.step_count >= self.max_steps
        poisoned    = was_adversarial and action_type == "delegate" and stakes >= CRITICAL_POISON_STAKES

        if all_done or budget_gone or poisoned:
            # Emit terminal reward on top of step reward
            return self._terminal_reward(
                step_reward=reward_value,
                step_reason=reason,
                step_breakdown=breakdown,
                forced_end=poisoned,
            )

        return self._build_step_result(reward_value, reason, breakdown, done=False)

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self, session_id: str | None = None) -> dict:
        if self.current_scenario is None:
            raise RuntimeError("No active episode. Call reset() first.")
        return {
            "episode_id":             self.episode_id,
            "session_id":             session_id or self.session_id,
            "step_count":             self.step_count,
            "max_steps":              self.max_steps,
            "total_reward":           round(self.total_reward, 4),
            "score":                  round(self.normalized_score(), 4),
            "done":                   self.done,
            "scenario_id":            self.current_scenario["scenario_id"],
            "task_type":              self.current_scenario["task_type"],
            "difficulty":             self._difficulty(),
            "status":                 self.episode_status,
            "last_reward":            round(self.last_reward, 4),
            "subtasks_completed":     self._graph.subtasks_completed(),
            "subtasks_total":         self._graph.subtasks_total(),
            "trust_snapshot":         self._ledger.snapshot(),
            "adversarial_detections": self._graph.adversarial_detections(),
            "adversarial_poisonings": self._graph.adversarial_poisonings(),
            "behavioral_fingerprints": self._ledger.behavioral_fingerprints(),
            "difficulty_profile":      self._difficulty_profile.to_dict(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grade_step(
        self,
        task_type: str,
        action_type: str,
        specialist_id: str | None,
        outcome: float,
        stakes: float,
        was_adversarial: bool,
        confidence: float | None,
        result_metadata: dict[str, Any],
        trust_score: float | None,
    ) -> tuple[float, str, dict]:

        if task_type == "task1":
            return grade_task1_step(
                chosen_specialist=specialist_id or "SELF",
                specialist_outcome=outcome,
                stakes=stakes,
                was_adversarial=was_adversarial,
                action_type=action_type,
                confidence=confidence,
                result_metadata=result_metadata,
                trust_score=trust_score,
            )
        elif task_type == "task2":
            return grade_task2_step(
                specialist_outcome=outcome,
                action_type=action_type,
                step_count=self.step_count,
                max_steps=self.max_steps,
                confidence=confidence,
                result_metadata=result_metadata,
            )
        else:  # task3
            return grade_task3_step(
                specialist_outcome=outcome,
                stakes=stakes,
                was_adversarial=was_adversarial,
                action_type=action_type,
                step_count=self.step_count,
                max_steps=self.max_steps,
                confidence=confidence,
                result_metadata=result_metadata,
                trust_score=trust_score,
            )

    def _terminal_reward(
        self,
        step_reward: float = 0.0,
        step_reason: str = "",
        step_breakdown: dict | None = None,
        forced_end: bool = False,
    ) -> dict:
        task_type = self.current_scenario["task_type"]

        if task_type == "task1":
            # Terminal for task1: completion-based if no step reward provided
            from graders import clamp
            if step_reward > 0.0:
                terminal_value    = step_reward
                terminal_reason   = step_reason
                terminal_breakdown = step_breakdown or {}
            else:
                completion        = self._graph.completion_rate()
                terminal_value    = clamp(0.4 + 0.5 * completion)
                terminal_reason   = f"Task1 complete. Completion={completion:.0%}."
                terminal_breakdown = {"completion_rate": round(completion, 3)}
        elif task_type == "task2":
            terminal_value, terminal_reason, terminal_breakdown = grade_task2_terminal(
                self._graph, self._ledger, self._public_ground_truth_reliability()
            )
        else:
            terminal_value, terminal_reason, terminal_breakdown = grade_task3_terminal(
                self._graph, self._ledger, self._public_ground_truth_reliability(),
                self.step_count, self.max_steps,
            )

        if forced_end:
            terminal_value  = 0.01
            terminal_reason = "Critical adversarial poison accepted at peak stakes — episode failed."

        self.last_reward    = terminal_value
        self.total_reward  += terminal_value
        self.reward_events += 1
        self.done           = True
        self.episode_status = "failed" if forced_end else "completed"
        self._record_reward_event(
            kind="terminal",
            action_type="terminal",
            specialist_id=None,
            subtask=None,
            stakes=0.0,
            reward_value=terminal_value,
            reason=terminal_reason,
            breakdown=terminal_breakdown,
            was_adversarial=False,
            confidence=None,
            result_metadata={},
            trust_before=None,
        )
        if self._difficulty_profile.adaptive:
            self._difficulty_controller.update(
                {
                    "adversarial_detections": self._graph.adversarial_detections(),
                    "adversarial_poisonings": self._graph.adversarial_poisonings(),
                    "adversarial_encounters": (
                        self._graph.adversarial_detections()
                        + self._graph.adversarial_poisonings()
                    ),
                }
            )

        return self._build_step_result(
            terminal_value, terminal_reason, terminal_breakdown,
            done=True,
            extra_info={
                **self._graph.summary(),
                "trust_snapshot": self._ledger.snapshot(),
                "forced_end":     forced_end,
                "difficulty_profile": self._difficulty_profile.to_dict(),
                "reward_report": self.reward_report(),
            },
        )

    def _build_step_result(
        self,
        reward_value: float,
        reason: str,
        breakdown: dict,
        done: bool,
        extra_info: dict | None = None,
    ) -> dict:
        node = self._graph.current_node() if self._graph and not done else None
        subtask_index = self._graph.node_index(node.subtask["id"]) if node else (
            self._graph.subtasks_total() if self._graph else 0
        )

        obs = {
            "session_id":            self.session_id,
            "scenario_id":           self.current_scenario["scenario_id"] if self.current_scenario else "",
            "task_type":             self.current_scenario["task_type"] if self.current_scenario else "",
            "difficulty":            self._difficulty(),
            "task_description":      self.current_scenario["description"] if self.current_scenario else "",
            "current_subtask":       node.subtask["description"] if node else "All subtasks complete.",
            "subtask_index":         subtask_index,
            "subtasks_total":        self._graph.subtasks_total() if self._graph else 0,
            "subtasks_remaining":    self._graph.subtasks_remaining() if self._graph else 0,
            "available_specialists": self._pool.available_ids(),
            "trust_snapshot":        self._ledger.snapshot(),
            "behavioral_fingerprints": self._ledger.behavioral_fingerprints(),
            "difficulty_profile":    self._difficulty_profile.to_dict(),
            "stakes_level":          node.subtask["stakes"] if node else 0.0,
            "step_count":            self.step_count,
            "max_steps":             self.max_steps,
            "last_action_summary":   self.last_action_summary,
            "last_reward":           round(self.last_reward, 4),
            "episode_status":        self.episode_status,
        }

        reward = {
            "value":            round(reward_value, 4),
            "reason":           reason,
            "signal_breakdown": breakdown,
        }

        info = {
            "episode_id":   self.episode_id,
            "session_id":   self.session_id,
            "step_count":   self.step_count,
            "max_steps":    self.max_steps,
            "total_reward": round(self.total_reward, 4),
            "score":        round(self.normalized_score(), 4),
        }
        if extra_info:
            info.update(extra_info)

        return {"observation": obs, "reward": reward, "done": done, "info": info}

    def _difficulty(self) -> str:
        return {"task1": "easy", "task2": "medium", "task3": "hard"}.get(
            self.current_scenario["task_type"] if self.current_scenario else "task3", "hard"
        )

    def normalized_score(self) -> float:
        """
        Mean reward over emitted grading events, normalized to 0.0-1.0.

        This is intentionally not a cumulative return. The terminal reward and
        efficiency terms already penalize unfinished or wasteful episodes, while
        this scalar stays comparable across tasks with different horizons.
        """
        if self.reward_events <= 0:
            return 0.0
        return max(0.0, min(1.0, self.total_reward / self.reward_events))

    def _public_ground_truth_reliability(self) -> dict[str, float]:
        return self._pool.public_ground_truth_reliability(_GROUND_TRUTH_RELIABILITY)

    def stream_snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
            "trust_snapshot": self._ledger.snapshot(),
            "behavioral_fingerprints": self._ledger.behavioral_fingerprints(),
            "difficulty_profile": self._difficulty_profile.to_dict(),
            "last_action_summary": self.last_action_summary,
            "last_reward": round(self.last_reward, 4),
        }

    def reward_report(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "task_type": self.current_scenario["task_type"] if self.current_scenario else "",
            "score": round(self.normalized_score(), 4),
            "total_reward": round(self.total_reward, 4),
            "reward_events": self.reward_events,
            "component_averages": self._reward_component_averages(),
            "events": list(self._reward_trace),
            "formula": {
                "task1_step": "0.43 accuracy + 0.30 stakes + 0.12 efficiency + 0.07 confidence + 0.04 domain + 0.04 verify",
                "task2_step": "0.55 accuracy + 0.25 efficiency + 0.10 confidence + 0.10 domain",
                "task3_step": "0.32 accuracy + 0.33 stakes + 0.10 efficiency + 0.10 confidence + 0.10 verify + 0.05 domain",
                "task3_terminal": "0.35 completion + 0.30 detection + 0.25 calibration + 0.10 efficiency",
            },
        }

    def _record_reward_event(
        self,
        kind: str,
        action_type: str,
        specialist_id: str | None,
        subtask: dict[str, Any] | None,
        stakes: float,
        reward_value: float,
        reason: str,
        breakdown: dict,
        was_adversarial: bool,
        confidence: float | None,
        result_metadata: dict[str, Any],
        trust_before: float | None,
    ) -> None:
        event = {
            "kind": kind,
            "step_count": self.step_count,
            "action_type": action_type,
            "specialist_id": specialist_id,
            "subtask_id": subtask.get("id") if subtask else None,
            "domain": subtask.get("domain") if subtask else None,
            "stakes": round(stakes, 3),
            "reward": round(reward_value, 4),
            "reason": reason,
            "signal_breakdown": breakdown,
            "was_adversarial": was_adversarial,
            "confidence": round(confidence, 3) if confidence is not None else None,
            "trust_before": round(trust_before, 3) if trust_before is not None else None,
            "trust_after": self._ledger.snapshot().get(specialist_id) if specialist_id else None,
            "result_metadata": result_metadata,
        }
        self._reward_trace.append(event)

    def _reward_component_averages(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for event in self._reward_trace:
            for key, value in event.get("signal_breakdown", {}).items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1
        return {
            key: round(total / max(1, counts[key]), 4)
            for key, total in sorted(totals.items())
        }

    def _apply_difficulty_profile(
        self,
        scenario: Scenario,
        profile: DifficultyProfile,
    ) -> Scenario:
        scenario_copy = copy.deepcopy(scenario)
        if not profile.adaptive or scenario_copy["task_type"] != "task3":
            return scenario_copy

        subtasks = scenario_copy["subtasks"]
        desired_high_stakes = max(1, round(len(subtasks) * profile.high_stakes_ratio))
        for offset, subtask in enumerate(subtasks[-desired_high_stakes:]):
            target_stakes = min(0.99, profile.adversarial_threshold + 0.05 + offset * 0.02)
            if subtask["stakes"] < target_stakes:
                subtask["stakes"] = round(target_stakes, 2)
                subtask["description"] = re.sub(
                    r"stakes=\d+\.\d+",
                    f"stakes={subtask['stakes']:.2f}",
                    subtask["description"],
                )
        return scenario_copy
