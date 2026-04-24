from __future__ import annotations

import time
import unittest

from app import SessionStore
from environment import SentinelEnv


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


if __name__ == "__main__":
    unittest.main()

