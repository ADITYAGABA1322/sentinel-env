from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class RewardClaim:
    agent_id: str
    reward: float
    justification: str
    timestamp: int
    approved: bool | None = None


@dataclass
class ActionRecord:
    agent_id: str
    action: dict[str, Any]
    timestamp: int


class AuditLedger:
    """
    Records reward claims and actions so an Auditor policy can detect hacking.
    """

    def __init__(self) -> None:
        self._claims: list[RewardClaim] = []
        self._actions: list[ActionRecord] = []
        self._flags: dict[str, list[str]] = {}

    def record_claim(
        self,
        agent_id: str,
        reward: float,
        justification: str,
        timestamp: int,
        approved: bool | None = None,
    ) -> None:
        self._claims.append(
            RewardClaim(
                agent_id=agent_id,
                reward=max(0.0, min(1.0, reward)),
                justification=justification,
                timestamp=timestamp,
                approved=approved,
            )
        )

    def record_action(self, agent_id: str, action: dict[str, Any], timestamp: int) -> None:
        self._actions.append(ActionRecord(agent_id=agent_id, action=dict(action), timestamp=timestamp))

    def flag(self, agent_id: str, reason: str) -> None:
        self._flags.setdefault(agent_id, []).append(reason)

    def clear(self, agent_id: str) -> None:
        self._flags.pop(agent_id, None)

    def anomaly_scores(self) -> dict[str, float]:
        """
        Simple deterministic anomaly score in [0, 1].

        High scores mean the agent's reward claims are unusually high, volatile,
        denied by the auditor, or already flagged.
        """
        scores: dict[str, float] = {}
        for agent_id in self._agent_ids():
            claims = [claim for claim in self._claims if claim.agent_id == agent_id]
            if not claims:
                scores[agent_id] = 0.0
                continue
            rewards = [claim.reward for claim in claims]
            avg_reward = statistics.fmean(rewards)
            volatility = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            denied_rate = sum(1 for claim in claims if claim.approved is False) / len(claims)
            high_claim_rate = sum(1 for reward in rewards if reward >= 0.90) / len(rewards)
            flag_bonus = min(0.25, 0.08 * len(self._flags.get(agent_id, [])))
            score = (
                0.35 * high_claim_rate
              + 0.25 * max(0.0, avg_reward - 0.70) / 0.30
              + 0.20 * min(1.0, volatility / 0.40)
              + 0.20 * denied_rate
              + flag_bonus
            )
            scores[agent_id] = round(max(0.0, min(1.0, score)), 4)
        return scores

    def investigate(self, agent_id: str, window: int = 10) -> dict[str, Any]:
        latest_time = self._latest_timestamp()
        since = max(0, latest_time - window)
        claims = [
            claim for claim in self._claims
            if claim.agent_id == agent_id and claim.timestamp >= since
        ]
        actions = [
            record for record in self._actions
            if record.agent_id == agent_id and record.timestamp >= since
        ]
        rewards = [claim.reward for claim in claims]
        return {
            "agent_id": agent_id,
            "window": window,
            "claims": [claim.__dict__ for claim in claims],
            "actions": [record.__dict__ for record in actions],
            "avg_claimed_reward": round(statistics.fmean(rewards), 4) if rewards else 0.0,
            "denied_claims": sum(1 for claim in claims if claim.approved is False),
            "flags": list(self._flags.get(agent_id, [])),
            "anomaly_score": self.anomaly_scores().get(agent_id, 0.0),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "claims": [claim.__dict__ for claim in self._claims],
            "actions": [record.__dict__ for record in self._actions],
            "anomaly_scores": self.anomaly_scores(),
            "flags": {agent: list(reasons) for agent, reasons in self._flags.items()},
        }

    def _agent_ids(self) -> set[str]:
        return (
            {claim.agent_id for claim in self._claims}
            | {record.agent_id for record in self._actions}
            | set(self._flags)
        )

    def _latest_timestamp(self) -> int:
        timestamps = [claim.timestamp for claim in self._claims] + [
            record.timestamp for record in self._actions
        ]
        return max(timestamps) if timestamps else 0
