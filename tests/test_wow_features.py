from __future__ import annotations

import unittest

from difficulty_controller import DifficultyController
from environment import SentinelEnv


class WowFeatureTests(unittest.TestCase):
    def test_difficulty_controller_tightens_after_strong_detection_window(self) -> None:
        controller = DifficultyController(window_size=2)

        controller.update({"adversarial_detections": 3, "adversarial_poisonings": 1})
        profile = controller.update({"adversarial_detections": 4, "adversarial_poisonings": 0})

        self.assertLess(profile.adversarial_threshold, 0.70)
        self.assertGreater(profile.high_stakes_ratio, 0.35)
        self.assertGreater(profile.verify_budget_penalty, 0)
        self.assertLess(profile.adversary_poison_confidence, 0.92)

    def test_observation_exposes_behavioral_fingerprints_without_hidden_identity(self) -> None:
        env = SentinelEnv()
        result = env.reset(task_type="task3", seed=42)
        obs = result["observation"]

        action = {
            "session_id": obs["session_id"],
            "task_type": "task3",
            "action_type": "delegate",
            "specialist_id": "S0",
        }
        result = env.step(action)
        fingerprints = result["observation"]["behavioral_fingerprints"]

        self.assertIn("S0", fingerprints)
        self.assertIn("confidence_accuracy_gap", fingerprints["S0"])
        self.assertIn("domain_hit_rate", fingerprints["S0"])
        self.assertNotIn("public_slot_to_internal_behavior", result["observation"])

    def test_adaptive_reset_adds_profile_to_observation(self) -> None:
        env = SentinelEnv()
        result = env.reset(task_type="task3", seed=42, adaptive=True)
        profile = result["observation"]["difficulty_profile"]

        self.assertTrue(profile["adaptive"])
        self.assertIn("adversarial_threshold", profile)


if __name__ == "__main__":
    unittest.main()
