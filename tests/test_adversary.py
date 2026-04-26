from __future__ import annotations

import unittest

from adversary import AdversaryFSM


class AdversaryFSMTests(unittest.TestCase):
    def test_attack_generation_and_detection_escalation(self) -> None:
        adversary = AdversaryFSM(seed=7, attack_probability=1.0, min_attack_gap=0)

        first = adversary.maybe_inject(
            step=1,
            auditor_detection_rate=0.0,
            gpu_ids=["GPU-00"],
            job_ids=["JOB-001"],
        )
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first["level"], 1)
        self.assertEqual(first["attack_type"], "false_completion")

        adversary.record_detection(first["attack_id"], detected=True)
        self.assertEqual(adversary.current_level(), 2)
        self.assertEqual(adversary.detection_rate(), 1.0)

        second = adversary.maybe_inject(
            step=2,
            auditor_detection_rate=0.0,
            gpu_ids=["GPU-00"],
            job_ids=["JOB-001"],
        )
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second["level"], 2)
        self.assertEqual(second["attack_type"], "false_memory_report")
        self.assertEqual(second["payload"]["target"], "GPU-00")


if __name__ == "__main__":
    unittest.main()
