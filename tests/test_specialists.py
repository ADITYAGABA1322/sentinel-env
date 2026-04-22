from __future__ import annotations

import random
import unittest

from specialists import DomainBoundSpecialist, SpecialistPool


class SpecialistTests(unittest.TestCase):
    def test_domain_bound_matches_abstract_analysis_and_verify_tasks(self) -> None:
        specialist = DomainBoundSpecialist()

        in_domain = specialist.execute("Analyze the inputs and identify the key pattern.", 0.2, random.Random(1))
        out_domain = specialist.execute("Execute the planned action and report the outcome.", 0.2, random.Random(1))

        self.assertTrue(in_domain.metadata["in_domain"])
        self.assertFalse(out_domain.metadata["in_domain"])
        self.assertEqual(in_domain.outcome, 1.0)
        self.assertEqual(out_domain.outcome, 0.0)

    def test_profile_shuffle_keeps_public_reliability_aligned(self) -> None:
        pool = SpecialistPool()
        pool.reset(seed=7)

        profile = pool.internal_profile()
        reliability = pool.public_ground_truth_reliability({"S0": 0.9, "S1": 0.6, "S2": 0.7, "S3": 0.15, "S4": 0.65})

        self.assertEqual(set(profile), {"S0", "S1", "S2", "S3", "S4"})
        self.assertEqual(set(reliability), {"S0", "S1", "S2", "S3", "S4"})
        self.assertEqual(profile[pool.adversarial_slot], "S3")
        self.assertEqual(reliability[pool.adversarial_slot], 0.15)


if __name__ == "__main__":
    unittest.main()
