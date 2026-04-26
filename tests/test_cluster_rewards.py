from __future__ import annotations

import unittest

from cluster_rewards import (
    ai_reliability_modifier,
    auditor_reward,
    global_cluster_reward,
    resource_manager_reward,
    task3_cluster_terminal,
)


class ClusterRewardTests(unittest.TestCase):
    def test_auditor_reward_penalizes_false_positives(self) -> None:
        clean_score, _ = auditor_reward(0.8, 0.0, 0.8)
        noisy_score, _ = auditor_reward(0.8, 0.6, 0.8)

        self.assertGreater(clean_score, noisy_score)

    def test_resource_manager_reward_penalizes_waste(self) -> None:
        efficient, _ = resource_manager_reward(0.85, 0.8, 0.05)
        wasteful, _ = resource_manager_reward(0.85, 0.8, 0.8)

        self.assertGreater(efficient, wasteful)

    def test_global_reward_collapses_with_cluster_health(self) -> None:
        healthy, _ = global_cluster_reward(
            {"orchestrator": 0.9, "resource_manager": 0.9, "auditor": 0.9, "worker": 0.9},
            cluster_health_score=1.0,
        )
        collapsed, _ = global_cluster_reward(
            {"orchestrator": 0.9, "resource_manager": 0.9, "auditor": 0.9, "worker": 0.9},
            cluster_health_score=0.0,
        )

        self.assertEqual(healthy, 0.9)
        self.assertEqual(collapsed, 0.01)

    def test_global_reward_is_reduced_by_ai_reliability_modifier(self) -> None:
        strong, _ = global_cluster_reward(
            {"orchestrator": 0.9, "resource_manager": 0.9, "auditor": 0.9, "worker": 0.9},
            cluster_health_score=1.0,
            reliability_modifier=1.0,
        )
        brittle, breakdown = global_cluster_reward(
            {"orchestrator": 0.9, "resource_manager": 0.9, "auditor": 0.9, "worker": 0.9},
            cluster_health_score=1.0,
            reliability_modifier=0.5,
        )

        self.assertEqual(breakdown["ai_reliability_modifier"], 0.5)
        self.assertGreater(strong, brittle)

    def test_ai_reliability_modifier_exposes_real_world_failure_signals(self) -> None:
        score, breakdown = ai_reliability_modifier(
            loop_avoidance=0.8,
            context_memory_score=0.7,
            hallucination_resistance=0.6,
            evaluation_freshness=1.0,
        )

        self.assertEqual(score, 0.75)
        self.assertIn("context_memory_score", breakdown)
        self.assertIn("hallucination_resistance", breakdown)

    def test_task3_terminal_is_multi_objective(self) -> None:
        strong, breakdown = task3_cluster_terminal(0.8, 0.9, 0.85, 0.75, 0.7)
        weak, _ = task3_cluster_terminal(0.8, 0.1, 0.1, 0.75, 0.7)

        self.assertIn("reward_hack_detection_rate", breakdown)
        self.assertGreater(strong, weak)


if __name__ == "__main__":
    unittest.main()
