from __future__ import annotations

import random
import uuid
from typing import Any

from graders import (
    grade_task1_step,
    grade_task2_step, grade_task2_terminal,
    grade_task3_step, grade_task3_terminal,
)
from scenarios import sample_scenario, get_scenario, Scenario
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
        self.last_reward: float = 0.0
        self.done: bool = False
        self.episode_status: str = "active"
        self.last_action_summary: str | None = None
        # Structured per-decision log for counterfactual grading (task3 terminal).
        self._episode_log: list[dict[str, Any]] = []

        self._graph: TaskGraph | None = None
        self._ledger: TrustLedger = TrustLedger()
        self._pool: SpecialistPool = SpecialistPool()
        self._rng: random.Random = random.Random()
        # Cross-episode behavioral memory (optional; used by SSE dashboard payloads).
        try:
            from behavioral_atlas import BehavioralAtlas

            self.atlas = BehavioralAtlas()
        except Exception:
            self.atlas = None

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        task_type: str | None = None,
        scenario_id: str | None = None,
        seed: int | None = None,
    ) -> dict:

        self._rng = random.Random(seed)

        # Select scenario
        if scenario_id:
            scenario = get_scenario(scenario_id)
        else:
            task = task_type or "task3"
            scenario = sample_scenario(task, seed=seed)

        self.current_scenario = scenario
        self.episode_id       = str(uuid.uuid4())
        self.session_id       = str(uuid.uuid4())
        self.step_count       = 0
        self.max_steps        = MAX_STEPS[scenario["task_type"]]
        self.total_reward     = 0.0
        self.last_reward      = 0.0
        self.done             = False
        self.episode_status   = "active"
        self.last_action_summary = None
        self._episode_log = []

        # Reset subcomponents
        self._graph = TaskGraph(scenario)
        self._ledger.reset()
        self._pool.reset(seed=seed)

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
        confidence = 0.5

        self.step_count += 1

        # --- Execute specialist or self-solve ---
        if action_type == "skip":
            self._graph.skip_node(subtask["id"])
            outcome        = 0.0
            was_adversarial = False
            self.last_action_summary = f"Skipped {subtask['id']}"

        elif action_type == "solve_independently":
            # Agent solves itself — always correct (no specialist involved)
            # But costs 2 steps (enforced via max_steps budget pressure)
            self.step_count += 1   # extra step cost
            outcome         = 1.0
            was_adversarial = False
            confidence = 0.5
            self._graph.record_outcome(subtask["id"], outcome, "SELF")
            self.last_action_summary = f"Self-solved {subtask['id']}"

        elif action_type == "verify":
            # First get result, then verify (costs +1 step)
            result = self._pool.execute(specialist_id, subtask["description"], stakes, self._rng)
            self.step_count += 1   # verification step cost
            outcome         = result.outcome if not result.is_adversarial else 0.0
            was_adversarial  = result.is_adversarial
            confidence = float(getattr(result, "confidence", 0.5))
            # Verification means agent caught adversarial — treat as detection
            if result.is_adversarial:
                outcome = 1.0   # successfully avoided
            self._graph.record_outcome(subtask["id"], outcome, specialist_id, was_adversarial)
            self._ledger.update(specialist_id, result.outcome, stakes)
            self.last_action_summary = f"Verified {specialist_id} on {subtask['id']}"

        else:  # delegate
            result          = self._pool.execute(specialist_id, subtask["description"], stakes, self._rng)
            was_adversarial  = result.is_adversarial
            outcome         = 0.0 if was_adversarial else result.outcome
            confidence = float(getattr(result, "confidence", 0.5))
            self._graph.record_outcome(subtask["id"], outcome, specialist_id, was_adversarial)
            self._ledger.update(specialist_id, result.outcome, stakes)
            self.last_action_summary = f"Delegated to {specialist_id} on {subtask['id']}"

        # Record a per-decision episode log entry for counterfactual replay.
        # This uses decision-step indexing (not step budget consumption).
        self._episode_log.append(
            {
                "step": len(self._episode_log) + 1,
                "action_type": str(action_type),
                "specialist_id": str(specialist_id) if specialist_id else None,
                "stakes": float(stakes),
                "was_correct": bool(outcome > 0.0),
                "confidence": float(confidence),
                "was_adversarial_attempt": bool(was_adversarial),
                "agent_caught_it": bool(action_type in ("verify", "solve_independently")),
            }
        )

        # --- Grade this step ---
        reward_value, reason, breakdown = self._grade_step(
            task_type, action_type, specialist_id, outcome,
            stakes, was_adversarial,
        )

        self.last_reward   = reward_value
        self.total_reward += reward_value

        # --- Check episode end ---
        all_done    = self._graph.is_done()
        budget_gone = self.step_count >= self.max_steps
        poisoned    = was_adversarial and action_type == "delegate" and stakes >= 0.85

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
    ) -> tuple[float, str, dict]:

        if task_type == "task1":
            return grade_task1_step(
                chosen_specialist=specialist_id or "SELF",
                specialist_outcome=outcome,
                stakes=stakes,
                was_adversarial=was_adversarial,
                action_type=action_type,
            )
        elif task_type == "task2":
            return grade_task2_step(
                specialist_outcome=outcome,
                action_type=action_type,
                step_count=self.step_count,
                max_steps=self.max_steps,
            )
        else:  # task3
            return grade_task3_step(
                specialist_outcome=outcome,
                stakes=stakes,
                was_adversarial=was_adversarial,
                action_type=action_type,
                step_count=self.step_count,
                max_steps=self.max_steps,
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
            gt_public = self._pool.public_ground_truth_reliability(_GROUND_TRUTH_RELIABILITY)
            terminal_value, terminal_reason, terminal_breakdown = grade_task2_terminal(
                self._graph, self._ledger, gt_public
            )
        else:
            gt_public = self._pool.public_ground_truth_reliability(_GROUND_TRUTH_RELIABILITY)
            terminal_value, terminal_reason, terminal_breakdown = grade_task3_terminal(
                self._graph,
                self._ledger,
                gt_public,
                self.step_count, self.max_steps,
                episode_log=list(self._episode_log),
                ground_truth_adversarial_slot=self._pool.adversarial_slot,
            )

        if forced_end:
            terminal_value  = 0.01
            terminal_reason = "Critical adversarial poison accepted at peak stakes — episode failed."

        self.last_reward    = terminal_value
        self.total_reward  += terminal_value
        self.done           = True
        self.episode_status = "failed" if forced_end else "completed"

        return self._build_step_result(
            terminal_value, terminal_reason, terminal_breakdown,
            done=True,
            extra_info={
                **self._graph.summary(),
                "trust_snapshot": self._ledger.snapshot(),
                "forced_end":     forced_end,
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

        obs = {
            "session_id":            self.session_id,
            "scenario_id":           self.current_scenario["scenario_id"] if self.current_scenario else "",
            "task_type":             self.current_scenario["task_type"] if self.current_scenario else "",
            "difficulty":            self._difficulty(),
            "task_description":      self.current_scenario["description"] if self.current_scenario else "",
            "current_subtask":       node.subtask["description"] if node else "All subtasks complete.",
            "subtask_index":         node.subtask["id"] if node else "DONE",
            "subtasks_total":        self._graph.subtasks_total() if self._graph else 0,
            "subtasks_remaining":    self._graph.subtasks_remaining() if self._graph else 0,
            "available_specialists": self._pool.available_ids(),
            "trust_snapshot":        self._ledger.snapshot(),
            "behavioral_fingerprints": self._compute_live_fingerprints(),
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
        }
        if extra_info:
            info.update(extra_info)

        return {"observation": obs, "reward": reward, "done": done, "info": info}

    def _compute_live_fingerprints(self) -> dict:
        """
        Computes 3 behavioral features per specialist from CURRENT episode history.
        These are the learnable signals for adversarial pattern recognition.
        """
        result: dict[str, dict[str, float]] = {}
        for slot in ["S0", "S1", "S2", "S3", "S4"]:
            slot_steps = [s for s in self._episode_log if s.get("specialist_id") == slot]
            if len(slot_steps) < 2:
                result[slot] = {"confidence_accuracy_gap": 0.0, "stakes_volatility": 0.0, "reuse_decay": 0.0}
                continue

            # Feature 1: confidence_accuracy_gap
            # High = specialist claims high confidence but is often wrong
            confidences = [float(s.get("confidence", 0.5) or 0.5) for s in slot_steps]
            accuracies = [float(bool(s.get("was_correct", False))) for s in slot_steps]
            gap = sum(abs(c - a) for c, a in zip(confidences, accuracies)) / len(slot_steps)

            # Feature 2: stakes_volatility
            low_acc = [float(bool(s["was_correct"])) for s in slot_steps if float(s.get("stakes", 0) or 0) < 0.5] or [0.5]
            high_acc = [float(bool(s["was_correct"])) for s in slot_steps if float(s.get("stakes", 0) or 0) >= 0.5] or [0.5]
            volatility = abs(sum(high_acc) / len(high_acc) - sum(low_acc) / len(low_acc))

            # Feature 3: reuse_decay
            if len(slot_steps) >= 3:
                mid = len(slot_steps) // 2
                first_half = [float(bool(s["was_correct"])) for s in slot_steps[:mid]] or [0.0]
                second_half = [float(bool(s["was_correct"])) for s in slot_steps[mid:]] or [0.0]
                decay = max(0.0, sum(first_half) / len(first_half) - sum(second_half) / len(second_half))
            else:
                decay = 0.0

            result[slot] = {
                "confidence_accuracy_gap": round(gap, 3),
                "stakes_volatility": round(volatility, 3),
                "reuse_decay": round(decay, 3),
            }
        return result

    def _difficulty(self) -> str:
        return {"task1": "easy", "task2": "medium", "task3": "hard"}.get(
            self.current_scenario["task_type"] if self.current_scenario else "task3", "hard"
        )