from __future__ import annotations

import unittest

from graders import clamp, grade_task3_step
from scenarios import get_scenario
from task_graph import TaskGraph


class GraderAndGraphTests(unittest.TestCase):
    def test_clamp_is_boundary_exclusive(self) -> None:
        self.assertEqual(clamp(-10), 0.01)
        self.assertEqual(clamp(10), 0.99)
        self.assertEqual(clamp(0.42), 0.42)

    def test_adversarial_verify_gets_detection_reward(self) -> None:
        reward, reason, breakdown = grade_task3_step(
            specialist_outcome=1.0,
            stakes=0.85,
            was_adversarial=True,
            action_type="verify",
            step_count=10,
            max_steps=45,
        )

        self.assertGreater(reward, 0.8)
        self.assertIn("Adversarial detected", reason)
        self.assertEqual(breakdown["stakes_awareness"], 0.99)

    def test_failed_nodes_are_retriable_then_resolved(self) -> None:
        graph = TaskGraph(get_scenario("SCN-TASK1-001"))
        node = graph.current_node()
        self.assertIsNotNone(node)

        graph.record_outcome(node.subtask["id"], 0.0, "S1")
        retry = graph.current_node()
        self.assertEqual(retry.subtask["id"], node.subtask["id"])

        graph.record_outcome(node.subtask["id"], 0.0, "S1")
        next_node = graph.current_node()
        self.assertIsNotNone(next_node)
        self.assertNotEqual(next_node.subtask["id"], node.subtask["id"])


if __name__ == "__main__":
    unittest.main()
