from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean

from sentinel_config import ADVERSARIAL_TRIGGER_STAKES


@dataclass
class DifficultyProfile:
    """Snapshot of the adaptive curriculum knobs for a new episode."""

    adaptive: bool = False
    episodes_seen: int = 0
    rolling_detection_rate: float = 0.0
    adversarial_threshold: float = ADVERSARIAL_TRIGGER_STAKES
    high_stakes_ratio: float = 0.35
    verify_budget_penalty: int = 0
    adversary_benign_confidence: float = 0.88
    adversary_poison_confidence: float = 0.92

    def to_dict(self) -> dict[str, float | int | bool]:
        payload = asdict(self)
        payload["rolling_detection_rate"] = round(self.rolling_detection_rate, 3)
        payload["adversarial_threshold"] = round(self.adversarial_threshold, 3)
        payload["high_stakes_ratio"] = round(self.high_stakes_ratio, 3)
        payload["adversary_benign_confidence"] = round(self.adversary_benign_confidence, 3)
        payload["adversary_poison_confidence"] = round(self.adversary_poison_confidence, 3)
        return payload


@dataclass
class DifficultyController:
    """
    Tiny self-improving curriculum controller.

    Every window of episodes, it watches adversarial detection rate. Strong
    policies get harder episodes; struggling policies get easier recovery.
    """

    window_size: int = 20
    threshold_step: float = 0.05
    high_stakes_step: float = 0.10
    min_threshold: float = 0.40
    max_threshold: float = 0.85
    min_high_stakes_ratio: float = 0.25
    max_high_stakes_ratio: float = 0.80
    max_verify_budget_penalty: int = 8
    _profile: DifficultyProfile = field(default_factory=DifficultyProfile)
    _episode_detection_rates: list[float] = field(default_factory=list)

    def profile(self, adaptive: bool) -> DifficultyProfile:
        if not adaptive:
            return DifficultyProfile(adaptive=False)
        profile = DifficultyProfile(**asdict(self._profile))
        profile.adaptive = True
        return profile

    def update(self, episode_metrics: dict[str, float | int]) -> DifficultyProfile:
        detections = int(episode_metrics.get("adversarial_detections", 0))
        poisonings = int(episode_metrics.get("adversarial_poisonings", 0))
        encounters = int(episode_metrics.get("adversarial_encounters", detections + poisonings))
        detection_rate = detections / max(1, encounters)

        self._episode_detection_rates.append(detection_rate)
        self._profile.episodes_seen += 1
        window = self._episode_detection_rates[-self.window_size :]
        self._profile.rolling_detection_rate = mean(window) if window else 0.0

        if len(self._episode_detection_rates) % self.window_size == 0:
            self._adapt_from_window(self._profile.rolling_detection_rate)

        return self.profile(adaptive=True)

    def reset(self) -> None:
        self._profile = DifficultyProfile()
        self._episode_detection_rates = []

    def state(self) -> dict[str, float | int | bool]:
        return self.profile(adaptive=True).to_dict()

    def _adapt_from_window(self, detection_rate: float) -> None:
        if detection_rate > 0.70:
            self._profile.adversarial_threshold -= self.threshold_step
            self._profile.high_stakes_ratio += self.high_stakes_step
            self._profile.verify_budget_penalty += 1
        elif detection_rate < 0.30:
            self._profile.adversarial_threshold += self.threshold_step
            self._profile.high_stakes_ratio -= self.high_stakes_step
            self._profile.verify_budget_penalty -= 1

        # Adversarial arms race: if the defender catches the adversary often,
        # the attacker starts earlier and lowers confidence to blend in.
        if detection_rate > 0.60:
            self._profile.adversary_benign_confidence -= 0.03
            self._profile.adversary_poison_confidence -= 0.03

        self._profile.adversarial_threshold = max(
            self.min_threshold,
            min(self.max_threshold, self._profile.adversarial_threshold),
        )
        self._profile.high_stakes_ratio = max(
            self.min_high_stakes_ratio,
            min(self.max_high_stakes_ratio, self._profile.high_stakes_ratio),
        )
        self._profile.verify_budget_penalty = max(
            0,
            min(self.max_verify_budget_penalty, self._profile.verify_budget_penalty),
        )
        self._profile.adversary_benign_confidence = max(
            0.60,
            min(0.88, self._profile.adversary_benign_confidence),
        )
        self._profile.adversary_poison_confidence = max(
            0.70,
            min(0.92, self._profile.adversary_poison_confidence),
        )


GLOBAL_DIFFICULTY_CONTROLLER = DifficultyController()
