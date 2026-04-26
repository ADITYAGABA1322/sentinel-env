from __future__ import annotations

import random
import unittest

from gpu_pool import GPUPool, GPUState


class GPUPoolTests(unittest.TestCase):
    def test_allocation_overload_preempt_and_false_report(self) -> None:
        pool = GPUPool(num_gpus=2, memory_per_gpu=80)

        self.assertTrue(pool.allocate("JOB-001", "GPU-00", 60))
        self.assertTrue(pool.allocate("JOB-002", "GPU-00", 30))

        hidden = pool.snapshot(include_hidden=True)[0]
        self.assertEqual(hidden["state"], GPUState.OVERLOADED.value)
        self.assertEqual(hidden["memory_used"], 90)

        pool.inject_false_report("GPU-00", {"state": "IDLE", "memory_free": 40})
        visible = pool.snapshot(include_hidden=False)[0]
        self.assertEqual(visible["state"], "IDLE")
        self.assertEqual(visible["memory_free"], 40)
        self.assertTrue(visible["report_tampered"])

        self.assertTrue(pool.preempt("JOB-002"))
        hidden = pool.snapshot(include_hidden=True)[0]
        self.assertEqual(hidden["state"], GPUState.ALLOCATED.value)
        self.assertEqual(hidden["memory_free"], 20)

    def test_failure_and_recovery_cycle(self) -> None:
        pool = GPUPool(num_gpus=1, memory_per_gpu=80, failure_probability=1.0, recovery_steps=2)
        pool.allocate("JOB-001", "GPU-00", 20)

        failed = pool.tick(rng=random.Random(0))
        self.assertEqual(failed, ["GPU-00"])
        self.assertEqual(pool.snapshot(include_hidden=True)[0]["state"], GPUState.FAILED.value)
        self.assertEqual(pool.cluster_health_score(), 0.0)

        pool.tick(rng=random.Random(0))
        self.assertEqual(pool.snapshot(include_hidden=True)[0]["state"], GPUState.RECOVERING.value)

        pool.tick(rng=random.Random(0))
        pool.tick(rng=random.Random(0))
        snapshot = pool.snapshot(include_hidden=True)[0]
        self.assertEqual(snapshot["state"], GPUState.IDLE.value)
        self.assertEqual(snapshot["jobs"], [])


if __name__ == "__main__":
    unittest.main()
