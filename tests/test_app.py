from __future__ import annotations

import time
import unittest

from app import SessionStore
from app import app
from environment import SentinelEnv
from fastapi.testclient import TestClient


class SessionStoreTests(unittest.TestCase):
    def test_session_store_evicts_expired_sessions(self) -> None:
        store = SessionStore(ttl_seconds=0, max_active=10)
        env = SentinelEnv()
        store.set("expired", env)

        time.sleep(0.001)

        self.assertIsNone(store.get("expired"))
        self.assertEqual(store.stats()["active_sessions"], 0)

    def test_session_store_evicts_lru_when_full(self) -> None:
        store = SessionStore(ttl_seconds=60, max_active=1)
        first = SentinelEnv()
        second = SentinelEnv()

        store.set("first", first)
        store.set("second", second)

        self.assertIsNone(store.get("first"))
        self.assertIs(store.get("second"), second)

    def test_reward_report_endpoint_returns_active_trace(self) -> None:
        client = TestClient(app)
        reset = client.post("/reset", json={"task_type": "task3", "seed": 42})
        self.assertEqual(reset.status_code, 200)
        payload = reset.json()
        sid = payload["info"]["session_id"]
        obs = payload["observation"]

        step = client.post(
            f"/step?session_id={sid}",
            json={
                "session_id": sid,
                "task_type": obs["task_type"],
                "action_type": "delegate",
                "specialist_id": "S0",
            },
        )
        self.assertEqual(step.status_code, 200)

        report = client.get(f"/reward-report?session_id={sid}")

        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.json()["reward_events"], 1)

    def test_cluster_mode_reset_step_state_and_report(self) -> None:
        client = TestClient(app)
        reset = client.post("/reset", json={"mode": "cluster", "task_type": "task3", "seed": 42})
        self.assertEqual(reset.status_code, 200)
        payload = reset.json()
        sid = payload["info"]["session_id"]
        obs = payload["observation"]

        self.assertEqual(payload["info"]["environment_mode"], "cluster")
        self.assertIn("gpu_pool", obs)
        self.assertIn("ai_failure_coverage", obs)

        step = client.post(
            f"/step?session_id={sid}",
            json={
                "session_id": sid,
                "action_type": "allocate",
                "job_id": obs["job_queue_summary"][0]["job_id"],
                "gpu_id": "GPU-00",
                "worker_id": "S0",
            },
        )
        self.assertEqual(step.status_code, 200)
        self.assertEqual(step.json()["info"]["environment_mode"], "cluster")

        state = client.get(f"/state?session_id={sid}")
        self.assertEqual(state.status_code, 200)
        self.assertIn("cluster", state.json())

        report = client.get(f"/reward-report?session_id={sid}")
        self.assertEqual(report.status_code, 200)
        self.assertIn("ai_failure_coverage", report.json())

    def test_cluster_task_prefix_enables_cluster_mode(self) -> None:
        client = TestClient(app)
        reset = client.post("/reset", json={"task_type": "cluster_task1", "seed": 7})
        self.assertEqual(reset.status_code, 200)
        payload = reset.json()

        self.assertEqual(payload["info"]["environment_mode"], "cluster")
        self.assertEqual(len(payload["observation"]["gpu_pool"]), 8)

    def test_cluster_dashboard_route_is_available(self) -> None:
        client = TestClient(app)
        response = client.get("/cluster-dashboard")

        self.assertEqual(response.status_code, 200)
        self.assertIn("SENTINEL Live Trust", response.text)
        self.assertIn("cluster health", response.text)


if __name__ == "__main__":
    unittest.main()
