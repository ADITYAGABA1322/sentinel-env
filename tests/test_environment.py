from __future__ import annotations

import unittest

from environment import SentinelEnv


class EnvironmentTests(unittest.TestCase):
    def test_reset_observation_has_integer_subtask_index(self) -> None:
        env = SentinelEnv()
        result = env.reset(task_type="task3", seed=3)

        self.assertEqual(result["observation"]["subtask_index"], 0)
        self.assertEqual(result["info"]["score"], 0.0)
        self.assertFalse(result["done"])

    def test_accurate_slow_public_slot_costs_two_steps(self) -> None:
        env = SentinelEnv()
        result = env.reset(task_type="task1", seed=11)
        slow_slot = next(
            public_id
            for public_id, internal_id in env._pool.internal_profile().items()
            if internal_id == "S0"
        )

        result = env.step({
            "session_id": result["observation"]["session_id"],
            "task_type": "task1",
            "action_type": "delegate",
            "specialist_id": slow_slot,
        })

        self.assertEqual(result["info"]["step_count"], 2)

    def test_verify_accurate_slow_costs_specialist_plus_verify_step(self) -> None:
        env = SentinelEnv()
        result = env.reset(task_type="task1", seed=11)
        slow_slot = next(
            public_id
            for public_id, internal_id in env._pool.internal_profile().items()
            if internal_id == "S0"
        )

        result = env.step({
            "session_id": result["observation"]["session_id"],
            "task_type": "task1",
            "action_type": "verify",
            "specialist_id": slow_slot,
        })

        self.assertEqual(result["info"]["step_count"], 3)

    def test_self_solve_finishes_long_task_with_normalized_score(self) -> None:
        env = SentinelEnv()
        result = env.reset(task_type="task3", seed=5)

        while not result["done"]:
            obs = result["observation"]
            result = env.step({
                "session_id": obs["session_id"],
                "task_type": obs["task_type"],
                "action_type": "solve_independently",
                "subtask_response": "SELF_SOLVED",
            })

        self.assertEqual(result["info"]["completion_rate"], 1.0)
        self.assertGreater(result["info"]["step_count"], 2)
        self.assertGreaterEqual(result["info"]["score"], 0.0)
        self.assertLessEqual(result["info"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
