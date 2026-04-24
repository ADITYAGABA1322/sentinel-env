from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sentinel_config import ADVERSARIAL_AWARENESS_STAKES

from scenarios import Scenario, SubTask


TaskStatus = Literal["pending", "ready", "in_progress", "completed", "failed", "skipped"]
SummaryValue = str | int | float | bool


# ---------------------------------------------------------------------------
# Node state
# ---------------------------------------------------------------------------

@dataclass
class TaskNode:
    subtask: SubTask
    status: TaskStatus = "pending"
    outcome: float = 0.0         # 1.0 correct | 0.5 partial | 0.0 wrong
    specialist_used: str = ""
    attempts: int = 0
    was_adversarial: bool = False
    adversarial_detection_count: int = 0
    adversarial_poisoning_count: int = 0


# ---------------------------------------------------------------------------
# TaskGraph
# Manages the DAG of subtasks for one episode.
# Tracks dependencies, determines which nodes are "ready" to execute,
# and records outcomes.
# ---------------------------------------------------------------------------

class TaskGraph:
    MAX_ATTEMPTS_PER_NODE = 2

    def __init__(self, scenario: Scenario) -> None:
        self._scenario   = scenario
        self._nodes: dict[str, TaskNode] = {}
        self._order: list[str] = []   # insertion order (for iteration)
        self._build(scenario["subtasks"])

    def _build(self, subtasks: list[SubTask]) -> None:
        for st in subtasks:
            self._nodes[st["id"]] = TaskNode(subtask=st)
            self._order.append(st["id"])

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def current_node(self) -> TaskNode | None:
        """
        Returns the first 'ready' node (all dependencies completed).
        Returns None if all nodes are done or none are unblocked yet.
        """
        for sid in self._order:
            node = self._nodes[sid]
            if node.status == "failed" and node.attempts < self.MAX_ATTEMPTS_PER_NODE:
                node.status = "ready"
            if node.status == "pending" and self._deps_met(sid):
                node.status = "ready"
            if node.status == "ready":
                return node
        return None

    def _deps_met(self, subtask_id: str) -> bool:
        """All dependencies must be resolved before downstream work starts."""
        deps = self._nodes[subtask_id].subtask["depends_on"]
        return all(
            self._is_dependency_resolved(dep)
            for dep in deps
            if dep in self._nodes
        )

    def _is_dependency_resolved(self, subtask_id: str) -> bool:
        node = self._nodes[subtask_id]
        if node.status in ("completed", "skipped"):
            return True
        return node.status == "failed" and node.attempts >= self.MAX_ATTEMPTS_PER_NODE

    def is_done(self) -> bool:
        return all(
            n.status in ("completed", "skipped")
            or (n.status == "failed" and n.attempts >= self.MAX_ATTEMPTS_PER_NODE)
            for n in self._nodes.values()
        )

    def completion_rate(self) -> float:
        completed = sum(1 for n in self._nodes.values() if n.status == "completed")
        return completed / len(self._nodes) if self._nodes else 0.0

    def adversarial_detections(self) -> int:
        """
        Count of high-stakes adversarial attempts that were avoided.
        Avoided = node was adversarial AND orchestrator chose VERIFY or SOLVE_INDEPENDENTLY.
        """
        return sum(
            n.adversarial_detection_count for n in self._nodes.values()
        )

    def adversarial_poisonings(self) -> int:
        """
        Count of adversarial results that slipped through unchecked.
        """
        return sum(
            n.adversarial_poisoning_count for n in self._nodes.values()
        )

    def subtasks_remaining(self) -> int:
        return sum(
            1 for n in self._nodes.values()
            if n.status in ("pending", "ready", "in_progress")
            or (n.status == "failed" and n.attempts < self.MAX_ATTEMPTS_PER_NODE)
        )

    def subtasks_completed(self) -> int:
        return sum(1 for n in self._nodes.values() if n.status == "completed")

    def subtasks_total(self) -> int:
        return len(self._nodes)

    def subtasks_failed(self) -> int:
        return sum(1 for n in self._nodes.values() if n.status == "failed")

    def node_index(self, subtask_id: str) -> int:
        return self._order.index(subtask_id)

    def high_stakes_nodes(self) -> list[TaskNode]:
        return [n for n in self._nodes.values() if n.subtask["stakes"] >= ADVERSARIAL_AWARENESS_STAKES]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        subtask_id: str,
        outcome: float,
        specialist_id: str,
        was_adversarial: bool = False,
    ) -> None:
        if subtask_id not in self._nodes:
            raise KeyError(f"Unknown subtask_id: {subtask_id}")
        node = self._nodes[subtask_id]
        node.outcome         = outcome
        node.specialist_used = specialist_id
        node.attempts        += 1
        node.was_adversarial  = node.was_adversarial or was_adversarial
        if was_adversarial and outcome > 0.0:
            node.adversarial_detection_count += 1
        elif was_adversarial:
            node.adversarial_poisoning_count += 1
        node.status = "completed" if outcome > 0.0 else "failed"

    def skip_node(self, subtask_id: str) -> None:
        if subtask_id in self._nodes:
            self._nodes[subtask_id].status = "skipped"

    # ------------------------------------------------------------------
    # Summary (for info dict in StepResult)
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, SummaryValue]:
        return {
            "scenario_id":          self._scenario["scenario_id"],
            "task_type":            self._scenario["task_type"],
            "subtasks_total":       self.subtasks_total(),
            "subtasks_completed":   self.subtasks_completed(),
            "subtasks_failed":      self.subtasks_failed(),
            "subtasks_remaining":   self.subtasks_remaining(),
            "completion_rate":      round(self.completion_rate(), 3),
            "adversarial_detections": self.adversarial_detections(),
            "adversarial_poisonings": self.adversarial_poisonings(),
            "is_done":              self.is_done(),
        }

    def node_statuses(self) -> dict[str, TaskStatus]:
        return {sid: n.status for sid, n in self._nodes.items()}
