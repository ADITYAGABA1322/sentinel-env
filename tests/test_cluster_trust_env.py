from __future__ import annotations

import unittest

from cluster_trust_env import ClusterTrustEnv


class ClusterTrustEnvTests(unittest.TestCase):
    def test_reset_returns_gpu_cluster_observation(self) -> None:
        env = ClusterTrustEnv()
        result = env.reset(task_type="task3", seed=42)
        obs = result["observation"]

        self.assertEqual(obs["task_type"], "task3")
        self.assertEqual(len(obs["gpu_pool"]), 16)
        self.assertEqual(len(obs["job_queue_summary"]), 30)
        self.assertEqual(obs["trust_snapshot"], {"S0": 0.5, "S1": 0.5, "S2": 0.5, "S3": 0.5, "S4": 0.5})
        self.assertIn("ai_failure_coverage", obs)
        self.assertTrue(obs["ai_failure_coverage"]["reward_hacking"]["covered"])

    def test_allocate_updates_gpu_jobs_trust_and_reward(self) -> None:
        env = ClusterTrustEnv()
        result = env.reset(task_type="task3", seed=42)
        obs = result["observation"]
        job_id = obs["job_queue_summary"][0]["job_id"]

        result = env.step(
            {
                "session_id": obs["session_id"],
                "action_type": "allocate",
                "job_id": job_id,
                "gpu_id": "GPU-00",
                "worker_id": "S0",
            }
        )
        state = env.state()

        self.assertFalse(result["done"])
        self.assertGreater(result["reward"]["value"], 0.0)
        self.assertEqual(state["jobs"]["statuses"]["running"], 1)
        self.assertGreaterEqual(state["cluster"]["memory_used"], 10)
        self.assertGreater(state["trust_snapshot"]["S0"], 0.5)

    def test_overreporter_false_report_lowers_worker_trust(self) -> None:
        env = ClusterTrustEnv()
        result = env.reset(task_type="task3", seed=3)
        obs = result["observation"]
        profile = env.state()["worker_profile_hidden"]
        overreporter = next(public_id for public_id, behavior in profile.items() if behavior == "overreporter")
        job_id = obs["job_queue_summary"][0]["job_id"]

        result = env.step(
            {
                "session_id": obs["session_id"],
                "action_type": "allocate",
                "job_id": job_id,
                "gpu_id": "GPU-00",
                "worker_id": overreporter,
            }
        )
        state = env.state()

        self.assertIn("trust_snapshot", state)
        self.assertLess(state["trust_snapshot"][overreporter], 0.5)
        self.assertGreaterEqual(result["reward"]["signal_breakdown"]["trust_calibration"], 0.0)

    def test_reward_report_contains_cluster_health_and_trust(self) -> None:
        env = ClusterTrustEnv()
        result = env.reset(task_type="task2", seed=9)
        obs = result["observation"]
        result = env.step({"session_id": obs["session_id"], "action_type": "allocate"})
        report = env.reward_report()

        self.assertEqual(report["reward_events"], 1)
        self.assertIn("cluster_health", report["events"][0])
        self.assertIn("trust_snapshot", report["events"][0])
        self.assertIn("global", report["events"][0]["signal_breakdown"])
        self.assertIn("ai_reliability", report["events"][0]["signal_breakdown"])

    def test_stream_snapshot_contains_live_dashboard_fields(self) -> None:
        env = ClusterTrustEnv()
        result = env.reset(task_type="task3", seed=42)
        obs = result["observation"]
        env.step({"session_id": obs["session_id"], "action_type": "allocate"})

        snapshot = env.stream_snapshot()

        self.assertEqual(snapshot["environment_mode"], "cluster")
        self.assertIn("cluster", snapshot)
        self.assertIn("jobs", snapshot)
        self.assertIn("ai_failure_coverage", snapshot)
        self.assertIn("attack_attempts", snapshot)

    def test_adaptive_reset_exposes_cluster_difficulty_profile(self) -> None:
        env = ClusterTrustEnv()
        result = env.reset(task_type="task3", seed=42, adaptive=True)
        obs = result["observation"]

        self.assertTrue(obs["difficulty_profile"]["adaptive"])
        self.assertIn("adversarial_threshold", obs["difficulty_profile"])

    def test_repeated_failed_actions_trigger_loop_and_context_signals(self) -> None:
        env = ClusterTrustEnv()
        result = env.reset(task_type="task3", seed=11)
        obs = result["observation"]

        for _ in range(4):
            result = env.step(
                {
                    "session_id": obs["session_id"],
                    "action_type": "verify",
                    "job_id": "missing-job",
                }
            )

        coverage = env.state()["ai_failure_coverage"]

        self.assertGreater(coverage["agent_loop_reliability"]["loop_events"], 0)
        self.assertGreater(coverage["context_memory_loss"]["drift_events"], 0)
        self.assertLess(coverage["ai_reliability_modifier"], 1.0)

    def test_accepted_reward_hack_reduces_global_modifier(self) -> None:
        env = ClusterTrustEnv()
        result = env.reset(task_type="task3", seed=42)

        for worker_id in ("S0", "S1"):
            obs = result["observation"]
            job = next(job for job in obs["job_queue_summary"] if job["status"] == "queued")
            gpu = max(obs["gpu_pool"], key=lambda row: row["memory_free"])
            result = env.step(
                {
                    "session_id": obs["session_id"],
                    "action_type": "allocate",
                    "job_id": job["job_id"],
                    "gpu_id": gpu["id"],
                    "worker_id": worker_id,
                }
            )

        breakdown = result["reward"]["signal_breakdown"]
        coverage = env.state()["ai_failure_coverage"]

        self.assertGreater(env.state()["attack_poisonings"], 0)
        self.assertLess(breakdown["reward_hack_resistance"], 1.0)
        self.assertLess(coverage["reward_hacking"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
