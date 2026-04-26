from __future__ import annotations

import unittest

from audit_ledger import AuditLedger


class AuditLedgerTests(unittest.TestCase):
    def test_anomaly_scores_reward_hacking_pattern(self) -> None:
        ledger = AuditLedger()
        ledger.record_action("resource_mgr", {"action_type": "allocate"}, timestamp=1)
        ledger.record_claim("resource_mgr", 0.98, "all jobs complete", timestamp=2, approved=False)
        ledger.record_claim("resource_mgr", 0.97, "all jobs complete again", timestamp=3, approved=False)
        ledger.flag("resource_mgr", "claimed completion without job progress")

        scores = ledger.anomaly_scores()
        report = ledger.investigate("resource_mgr", window=5)

        self.assertGreater(scores["resource_mgr"], 0.65)
        self.assertEqual(report["denied_claims"], 2)
        self.assertEqual(report["flags"], ["claimed completion without job progress"])
        self.assertEqual(report["actions"][0]["agent_id"], "resource_mgr")


if __name__ == "__main__":
    unittest.main()
