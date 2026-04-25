from __future__ import annotations

from sentinel_config import ADVERSARIAL_AWARENESS_STAKES


class TrustLedger:
    """
    Bayesian reliability tracker for each specialist.

    Each specialist gets a Beta distribution prior (alpha, beta).
    alpha = successes + 1, beta = failures + 1 (Laplace smoothing).
    Trust score = alpha / (alpha + beta) = mean of Beta distribution.

    Stakes multiplier: high-stakes outcomes move the needle harder.
    Profile shuffles every episode — ledger resets on reset().
    """

    SPECIALIST_IDS = ["S0", "S1", "S2", "S3", "S4"]

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        # Uniform prior: alpha=1, beta=1 → trust=0.5 for all specialists
        self._alpha: dict[str, float] = {sid: 1.0 for sid in self.SPECIALIST_IDS}
        self._beta:  dict[str, float] = {sid: 1.0 for sid in self.SPECIALIST_IDS}
        self._call_count: dict[str, int] = {sid: 0 for sid in self.SPECIALIST_IDS}
        self._confidence_gap_sum: dict[str, float] = {sid: 0.0 for sid in self.SPECIALIST_IDS}
        self._confidence_count: dict[str, int] = {sid: 0 for sid in self.SPECIALIST_IDS}
        self._domain_success: dict[str, dict[str, float]] = {sid: {} for sid in self.SPECIALIST_IDS}
        self._domain_count: dict[str, dict[str, int]] = {sid: {} for sid in self.SPECIALIST_IDS}
        self._stakes_success: dict[str, dict[str, float]] = {
            sid: {"low": 0.0, "high": 0.0} for sid in self.SPECIALIST_IDS
        }
        self._stakes_count: dict[str, dict[str, int]] = {
            sid: {"low": 0, "high": 0} for sid in self.SPECIALIST_IDS
        }

    def reset(self) -> None:
        """Call at the start of each episode."""
        self._reset()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        specialist_id: str,
        outcome: float,   # 1.0 = correct, 0.0 = wrong/adversarial, 0.5 = partial
        stakes: float,    # 0.0–1.0; high stakes = larger update
        confidence: float | None = None,
        domain: str | None = None,
    ) -> None:
        """
        Bayesian update after observing a specialist outcome.
        stakes acts as a weight multiplier (1x at low stakes, 3x at high stakes).
        """
        if specialist_id not in self._alpha:
            return

        weight = 1.0 + 2.0 * stakes   # 1.0 → 3.0

        self._call_count[specialist_id] += 1

        if outcome >= 0.5:
            self._alpha[specialist_id] += weight * outcome
        else:
            self._beta[specialist_id] += weight * (1.0 - outcome)

        if confidence is not None:
            self._confidence_gap_sum[specialist_id] += max(0.0, confidence - outcome)
            self._confidence_count[specialist_id] += 1

        if domain:
            domain_key = domain.upper()
            self._domain_success[specialist_id][domain_key] = (
                self._domain_success[specialist_id].get(domain_key, 0.0) + outcome
            )
            self._domain_count[specialist_id][domain_key] = (
                self._domain_count[specialist_id].get(domain_key, 0) + 1
            )

        stakes_bucket = "high" if stakes >= ADVERSARIAL_AWARENESS_STAKES else "low"
        self._stakes_success[specialist_id][stakes_bucket] += outcome
        self._stakes_count[specialist_id][stakes_bucket] += 1

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def trust(self, specialist_id: str) -> float:
        """Point estimate: mean of Beta distribution."""
        a = self._alpha.get(specialist_id, 1.0)
        b = self._beta.get(specialist_id, 1.0)
        return a / (a + b)

    def snapshot(self) -> dict[str, float]:
        """Rounded trust scores for all specialists."""
        return {sid: round(self.trust(sid), 3) for sid in self.SPECIALIST_IDS}

    def behavioral_fingerprints(self) -> dict[str, dict]:
        """
        Public behavioral features an orchestrator can learn from.

        These are still evidence-only: no hidden specialist identity leaks.
        """
        fingerprints: dict[str, dict] = {}
        for sid in self.SPECIALIST_IDS:
            confidence_count = self._confidence_count[sid]
            gap = (
                self._confidence_gap_sum[sid] / confidence_count
                if confidence_count
                else 0.0
            )
            domain_hit_rate = {
                domain: round(success / max(1, self._domain_count[sid][domain]), 3)
                for domain, success in sorted(self._domain_success[sid].items())
            }
            low_rate = self._bucket_rate(sid, "low")
            high_rate = self._bucket_rate(sid, "high")
            volatility = abs(high_rate - low_rate) if low_rate is not None and high_rate is not None else 0.0
            fingerprints[sid] = {
                "calls": self._call_count[sid],
                "confidence_accuracy_gap": round(gap, 3),
                "domain_hit_rate": domain_hit_rate,
                "stakes_volatility": round(volatility, 3),
                "low_stakes_accuracy": round(low_rate, 3) if low_rate is not None else None,
                "high_stakes_accuracy": round(high_rate, 3) if high_rate is not None else None,
            }
        return fingerprints

    def _bucket_rate(self, specialist_id: str, bucket: str) -> float | None:
        count = self._stakes_count[specialist_id][bucket]
        if count == 0:
            return None
        return self._stakes_success[specialist_id][bucket] / count

    def call_count(self, specialist_id: str) -> int:
        return self._call_count.get(specialist_id, 0)

    def most_trusted(self) -> str:
        """Returns the specialist_id with the highest current trust score."""
        return max(self.SPECIALIST_IDS, key=self.trust)

    def least_trusted(self) -> str:
        return min(self.SPECIALIST_IDS, key=self.trust)

    # ------------------------------------------------------------------
    # Calibration score (used in reward engine)
    # ------------------------------------------------------------------

    def brier_score(self, ground_truth_reliability: dict[str, float]) -> float:
        """
        Measures how well the trust scores predict actual specialist reliability.
        Lower = better calibrated. Range 0.0–1.0.

        ground_truth_reliability: {"S0": 0.9, "S1": 0.6, ...}
        (hidden from agent, used only by reward engine)
        """
        total = 0.0
        n = 0
        for sid in self.SPECIALIST_IDS:
            if sid in ground_truth_reliability:
                predicted = self.trust(sid)
                actual    = ground_truth_reliability[sid]
                total += (predicted - actual) ** 2
                n += 1
        return total / n if n > 0 else 0.0

    def calibration_reward(self, ground_truth_reliability: dict[str, float]) -> float:
        """
        Convert Brier score to a reward signal (0.0–1.0).
        Perfect calibration → 1.0. Random → ~0.5.
        """
        brier = self.brier_score(ground_truth_reliability)
        # Invert and scale: brier=0 → reward=1.0, brier=0.25 → reward=0.5
        return max(0.0, 1.0 - 4.0 * brier)

    def __repr__(self) -> str:
        snap = self.snapshot()
        return f"TrustLedger({snap})"
