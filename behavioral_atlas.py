"""
behavioral_atlas.py
===================

Implements the **BehavioralAtlas**: a persistent, cross-episode memory of
specialist *behavioral archetypes* rather than specialist identities.

Why this exists
--------------
In SENTINEL, public slot labels (`S0`–`S4`) shuffle their hidden behaviors each
episode. Memorizing "S3 is adversarial" is not transferable. But behavioral
patterns are invariant and can be compressed into a small set of stable signals:

- **confidence_accuracy_gap**: confidence minus realized accuracy
- **stakes_volatility**: how sharply accuracy drops at high stakes
- **domain_consistency**: how consistent accuracy is across domains
- **temporal_decay_rate**: how quickly accuracy decays with repeated use

The atlas stores these compressed fingerprints and:
- **observe()**: compresses an episode trace to a `BehavioralFingerprint`
- **link_entities()**: tags fingerprints into threshold-defined archetypes and
  builds centroid fingerprints per archetype
- **seed_priors()**: produces trust priors for the next episode
- **save()/load()**: persists state across process restarts

This is conceptually inspired by mem0-style entity linking and semantic
compression, but specialized to behavioral fingerprinting under identity
shuffle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import math
import statistics


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _safe_mean(xs: List[float], default: float = 0.0) -> float:
    return float(sum(xs) / len(xs)) if xs else float(default)


def _cosine_similarity(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """
    Cosine similarity for 4D vectors.
    Returns 0.0 if either vector is effectively zero-norm.
    """
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    dot = ax * bx + ay * by + az * bz + aw * bw
    na = math.sqrt(ax * ax + ay * ay + az * az + aw * aw)
    nb = math.sqrt(bx * bx + by * by + bz * bz + bw * bw)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


@dataclass
class BehavioralFingerprint:
    """
    Compressed behavioral signature extracted after one episode.

    The fingerprint intentionally does **not** encode the hidden identity of a
    specialist. It describes *how* a specialist behaved within an episode, in a
    way that can be compared and clustered across episodes under identity
    shuffling.
    """

    # Key adversarial signal: reported confidence vs actual accuracy
    confidence_accuracy_gap: float  # high => overconfident or adversarial
    # How much accuracy changes as stakes rise (0.0 = stable, 1.0 = drops sharply)
    stakes_volatility: float
    # Consistency across domains (1.0 = consistent, 0.0 = domain-bound)
    domain_consistency: float
    # Accuracy decay over repeated calls (0.0 = stable, 1.0 = fast degrading)
    temporal_decay_rate: float

    # Raw outcome record for this episode (1.0 correct, 0.0 incorrect).
    outcomes: List[float] = field(default_factory=list)
    # Stakes at each step (0.0–1.0).
    stakes_at_steps: List[float] = field(default_factory=list)
    # Public slot label observed this episode (S0–S4) — NOT hidden identity.
    slot_label: str = ""
    # Episode-level reward for context/debugging.
    episode_reward: float = 0.0

    def vector(self) -> Tuple[float, float, float, float]:
        """
        Vector used for similarity comparisons.

        Note: `domain_consistency` is a "goodness" feature (higher is better),
        while the others are "risk" features (higher is worse). For similarity
        we keep the raw fingerprint values; for trust-worthiness we invert
        domain consistency where appropriate.
        """
        return (
            float(self.confidence_accuracy_gap),
            float(self.stakes_volatility),
            float(self.domain_consistency),
            float(self.temporal_decay_rate),
        )


class BehavioralAtlas:
    """
    Persistent cross-episode memory of specialist behavioral archetypes.

    Inspired by mem0's entity-linking architecture:
    - **observe()**: compress episode observations into a fingerprint
    - **link_entities()**: cluster/tag similar fingerprints into archetypes
    - **seed_priors()**: retrieve matching history to initialize trust priors

    The atlas stores behavioral fingerprints, not specialist identities.
    This makes it *identity-shuffle-invariant*: learned archetypes transfer even
    when the hidden behavior behind `S0`–`S4` permutes each episode.
    """

    def __init__(self, max_fingerprints: int = 500):
        # All stored fingerprints (across all episodes and all public slots)
        self.fingerprints: List[BehavioralFingerprint] = []
        # Discovered archetypal clusters {archetype_name: centroid_fingerprint}
        self.archetypes: Dict[str, BehavioralFingerprint] = {}
        self.max_fingerprints = int(max_fingerprints)
        self.episode_count = 0

        # Parallel metadata (persisted) for per-fingerprint archetype tags.
        self._fingerprint_tags: List[str] = []

    # ---------------------------------------------------------------------
    # Observation compression
    # ---------------------------------------------------------------------

    def observe(self, slot_label: str, step_history: List[dict], episode_reward: float) -> BehavioralFingerprint:
        """
        Compress one episode's interaction trace for a *single public slot*
        (e.g. 'S2') into a `BehavioralFingerprint`, store it, and return it.

        Parameters
        ----------
        slot_label:
            The public slot label (`S0`–`S4`) used this episode. This is *not*
            a stable identity across episodes.

        step_history:
            List of per-step dicts, each like:

            - `confidence`: float in [0, 1]
            - `was_correct`: bool
            - `stakes`: float in [0, 1]
            - `step_index`: int (monotonic for this episode/slot)
            - `domain`: str (may be missing/None)

        episode_reward:
            Episode-level reward, stored for context/analysis/debugging.

        Fingerprint dimensions (computed from `step_history`)
        -----------------------------------------------
        - confidence_accuracy_gap:
            Mean(confidence) - mean(accuracy), clamped to [0, 1]. High values
            indicate systematic overconfidence.

        - stakes_volatility:
            How sharply accuracy drops from low-stakes to high-stakes steps.
            Computed as clamp(low_stakes_acc - high_stakes_acc, 0, 1), with
            low stakes defined as stakes <= 0.4 and high stakes as stakes >= 0.7.

        - domain_consistency:
            1 - normalized stdev of per-domain accuracy means. Returns 1.0 if
            there is only one observed domain. (0.0 indicates strong domain bound.)

        - temporal_decay_rate:
            Accuracy decay over repeated calls within the episode.
            Computed as clamp(early_acc - late_acc, 0, 1), splitting the trace
            into an early and late half by step order.
        """
        slot = str(slot_label)
        if not step_history:
            fp = BehavioralFingerprint(
                confidence_accuracy_gap=0.0,
                stakes_volatility=0.0,
                domain_consistency=1.0,
                temporal_decay_rate=0.0,
                outcomes=[],
                stakes_at_steps=[],
                slot_label=slot,
                episode_reward=float(episode_reward),
            )
            self._store_fingerprint(fp, tag="unknown")
            return fp

        confidences: List[float] = []
        outcomes: List[float] = []
        stakes_list: List[float] = []
        domains: List[str] = []

        # Normalize and extract
        for ev in step_history:
            conf = _clamp(float(ev.get("confidence", 0.5)))
            was_correct = bool(ev.get("was_correct", False))
            stakes = _clamp(float(ev.get("stakes", 0.0)))
            domain = ev.get("domain")
            domain = str(domain) if domain is not None else "unknown"

            confidences.append(conf)
            outcomes.append(1.0 if was_correct else 0.0)
            stakes_list.append(stakes)
            domains.append(domain)

        mean_conf = _safe_mean(confidences, default=0.5)
        mean_acc = _safe_mean(outcomes, default=0.0)
        confidence_accuracy_gap = _clamp(mean_conf - mean_acc, 0.0, 1.0)

        # Stakes volatility: compare low vs high stakes accuracy bands
        low_accs = [o for o, s in zip(outcomes, stakes_list) if s <= 0.4]
        high_accs = [o for o, s in zip(outcomes, stakes_list) if s >= 0.7]
        low_acc = _safe_mean(low_accs, default=mean_acc)
        high_acc = _safe_mean(high_accs, default=mean_acc)
        stakes_volatility = _clamp(low_acc - high_acc, 0.0, 1.0)

        # Domain consistency: stdev of per-domain accuracy means
        by_domain: Dict[str, List[float]] = {}
        for d, o in zip(domains, outcomes):
            by_domain.setdefault(d, []).append(o)
        if len(by_domain) <= 1:
            domain_consistency = 1.0
        else:
            domain_means = [_safe_mean(v, default=0.0) for v in by_domain.values()]
            stdev = statistics.pstdev(domain_means) if len(domain_means) >= 2 else 0.0
            # Max stdev for means in [0,1] is 0.5 (e.g., [0,1]).
            domain_consistency = _clamp(1.0 - _clamp(stdev / 0.5, 0.0, 1.0), 0.0, 1.0)

        # Temporal decay: early-half accuracy - late-half accuracy
        n = len(outcomes)
        split = max(1, n // 2)
        early_acc = _safe_mean(outcomes[:split], default=mean_acc)
        late_acc = _safe_mean(outcomes[split:], default=mean_acc)
        temporal_decay_rate = _clamp(early_acc - late_acc, 0.0, 1.0)

        fp = BehavioralFingerprint(
            confidence_accuracy_gap=float(confidence_accuracy_gap),
            stakes_volatility=float(stakes_volatility),
            domain_consistency=float(domain_consistency),
            temporal_decay_rate=float(temporal_decay_rate),
            outcomes=list(outcomes),
            stakes_at_steps=list(stakes_list),
            slot_label=slot,
            episode_reward=float(episode_reward),
        )

        tag = self._tag_fingerprint(fp)
        self._store_fingerprint(fp, tag=tag)
        return fp

    def _store_fingerprint(self, fp: BehavioralFingerprint, tag: str) -> None:
        self.fingerprints.append(fp)
        self._fingerprint_tags.append(str(tag))
        # Enforce bounded memory
        if len(self.fingerprints) > self.max_fingerprints:
            overflow = len(self.fingerprints) - self.max_fingerprints
            if overflow > 0:
                self.fingerprints = self.fingerprints[overflow:]
                self._fingerprint_tags = self._fingerprint_tags[overflow:]
        self.episode_count += 1

    # ---------------------------------------------------------------------
    # Archetype tagging / linking
    # ---------------------------------------------------------------------

    def _tag_fingerprint(self, fp: BehavioralFingerprint) -> str:
        """
        Apply the threshold-based archetype rules (no ML) and return a tag.

        The rule thresholds are implemented exactly as specified:
        - adversarial_archetype: stakes_volatility > 0.6 AND confidence_accuracy_gap > 0.3
        - overconfident_archetype: confidence_accuracy_gap > 0.35 AND stakes_volatility < 0.3
        - degrading_archetype: temporal_decay_rate > 0.5
        - domain_bound_archetype: domain_consistency < 0.4
        - reliable_archetype: all four dimensions "low" (interpreting domain-consistency as low risk)
        """
        if fp.stakes_volatility > 0.6 and fp.confidence_accuracy_gap > 0.3:
            return "adversarial_archetype"
        if fp.confidence_accuracy_gap > 0.35 and fp.stakes_volatility < 0.3:
            return "overconfident_archetype"
        if fp.temporal_decay_rate > 0.5:
            return "degrading_archetype"
        if fp.domain_consistency < 0.4:
            return "domain_bound_archetype"

        # "reliable_archetype: all four dimensions low"
        # Domain consistency is a *goodness* feature (higher is better), so we
        # consider its risk as (1 - domain_consistency).
        gap = fp.confidence_accuracy_gap
        vol = fp.stakes_volatility
        dom_risk = 1.0 - fp.domain_consistency
        dec = fp.temporal_decay_rate
        if gap < 0.2 and vol < 0.2 and dom_risk < 0.2 and dec < 0.2:
            return "reliable_archetype"

        return "mixed"

    def link_entities(self):
        """
        Cluster/tag stored fingerprints into archetypes and update centroids.

        This method is designed to be called periodically (e.g., every 20
        episodes) rather than every episode.

        Implementation notes
        --------------------
        - Tagging uses the threshold-based archetype rules exactly as described.
        - Centroids are computed as the mean of the member fingerprints'
          dimensions (confidence gap, volatility, domain consistency, decay).
        - Cosine similarity is used internally for centroid stability and to
          keep behavior consistent with the "entity-linking" metaphor, but the
          rule-based tags are the authoritative assignment mechanism.
        """
        if not self.fingerprints:
            self.archetypes = {}
            return

        # Ensure tags are up to date (in case observe() was bypassed or tags loaded)
        if len(self._fingerprint_tags) != len(self.fingerprints):
            self._fingerprint_tags = [self._tag_fingerprint(fp) for fp in self.fingerprints]

        members: Dict[str, List[BehavioralFingerprint]] = {}
        for fp, tag in zip(self.fingerprints, self._fingerprint_tags):
            if tag == "mixed" or tag == "unknown":
                continue
            members.setdefault(tag, []).append(fp)

        new_archetypes: Dict[str, BehavioralFingerprint] = {}
        for name, fps in members.items():
            if not fps:
                continue
            centroid = BehavioralFingerprint(
                confidence_accuracy_gap=_safe_mean([f.confidence_accuracy_gap for f in fps]),
                stakes_volatility=_safe_mean([f.stakes_volatility for f in fps]),
                domain_consistency=_safe_mean([f.domain_consistency for f in fps]),
                temporal_decay_rate=_safe_mean([f.temporal_decay_rate for f in fps]),
                outcomes=[],
                stakes_at_steps=[],
                slot_label="CENTROID",
                episode_reward=_safe_mean([f.episode_reward for f in fps], default=0.0),
            )

            # If previous centroid exists, keep whichever is more "central" under cosine similarity.
            prev = self.archetypes.get(name)
            if prev is not None:
                prev_sim = _safe_mean([_cosine_similarity(prev.vector(), f.vector()) for f in fps], default=0.0)
                new_sim = _safe_mean([_cosine_similarity(centroid.vector(), f.vector()) for f in fps], default=0.0)
                if prev_sim >= new_sim:
                    centroid = prev

            new_archetypes[name] = centroid

        self.archetypes = new_archetypes

    # ---------------------------------------------------------------------
    # Priors seeding
    # ---------------------------------------------------------------------

    def _trustworthiness(self, fp: BehavioralFingerprint) -> float:
        """
        Map a fingerprint to a trust-worthiness scalar in [0, 1].

        Interpretation:
        - Higher confidence gap is worse
        - Higher stakes volatility is worse
        - Lower domain consistency is worse (so we use domain_risk = 1 - domain_consistency)
        - Higher temporal decay is worse
        """
        gap = _clamp(fp.confidence_accuracy_gap, 0.0, 1.0)
        vol = _clamp(fp.stakes_volatility, 0.0, 1.0)
        dom_risk = _clamp(1.0 - fp.domain_consistency, 0.0, 1.0)
        dec = _clamp(fp.temporal_decay_rate, 0.0, 1.0)
        risk = 0.30 * gap + 0.30 * vol + 0.20 * dom_risk + 0.20 * dec
        return _clamp(1.0 - risk, 0.0, 1.0)

    def seed_priors(self, n_specialists: int = 5) -> Dict[str, float]:
        """
        Compute initial trust priors for the start of a new episode.

        Returns a dict of public slot labels (`S0`..`S{n-1}`) to trust weights.

        Logic (as required)
        -------------------
        - If the atlas is empty: return uniform priors at 0.5 for each slot.
        - Otherwise, for each slot:
          - Find the **last-seen** fingerprint for that slot label.
          - Retrieve historical fingerprints that most resemble it (cosine sim).
          - Compute a similarity-weighted mean trust-worthiness over those matches.
          - Map trust-worthiness to the (0.1, 0.9) range via: 0.1 + 0.8 * score.
        - If adversarial archetype detected in >30% of fingerprints *for that slot*:
          return 0.25 (warranted suspicion).
        - If reliable archetype detected in >60% of fingerprints *for that slot*:
          return 0.72.

        Notes
        -----
        The atlas is identity-shuffle-invariant in the sense that it stores and
        compares behavioral patterns. The returned priors are per *public slot*
        label, because the environment API consumes slot-labeled trust vectors.
        """
        n = int(n_specialists)
        slots = [f"S{i}" for i in range(n)]

        if not self.fingerprints:
            return {s: 0.5 for s in slots}

        if len(self._fingerprint_tags) != len(self.fingerprints):
            self._fingerprint_tags = [self._tag_fingerprint(fp) for fp in self.fingerprints]

        # Index fingerprints by slot label
        fps_by_slot: Dict[str, List[Tuple[BehavioralFingerprint, str]]] = {s: [] for s in slots}
        for fp, tag in zip(self.fingerprints, self._fingerprint_tags):
            if fp.slot_label in fps_by_slot:
                fps_by_slot[fp.slot_label].append((fp, tag))

        priors: Dict[str, float] = {}
        for slot in slots:
            slot_items = fps_by_slot.get(slot) or []
            if not slot_items:
                priors[slot] = 0.5
                continue

            # Archetype override checks (slot-local)
            tags = [t for _, t in slot_items if t not in ("unknown",)]
            total = max(1, len(tags))
            adv_frac = sum(1 for t in tags if t == "adversarial_archetype") / total
            rel_frac = sum(1 for t in tags if t == "reliable_archetype") / total
            if adv_frac > 0.30:
                priors[slot] = 0.25
                continue
            if rel_frac > 0.60:
                priors[slot] = 0.72
                continue

            # Similarity-weighted trust based on last-seen behavior of this slot.
            last_fp = slot_items[-1][0]
            anchor = last_fp.vector()

            scored: List[Tuple[float, BehavioralFingerprint]] = []
            for fp, _tag in slot_items:
                sim = _clamp(_cosine_similarity(anchor, fp.vector()), 0.0, 1.0)
                scored.append((sim, fp))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_k = scored[: min(25, len(scored))]

            # Similarity-weighted average trust-worthiness; fall back to unweighted mean.
            weights = [s for s, _ in top_k if s > 1e-6]
            if weights:
                tw = sum(sim * self._trustworthiness(fp) for sim, fp in top_k) / sum(weights)
            else:
                tw = _safe_mean([self._trustworthiness(fp) for _, fp in top_k], default=0.5)

            prior = 0.1 + 0.8 * _clamp(tw, 0.0, 1.0)
            priors[slot] = float(_clamp(prior, 0.1, 0.9))

        return priors

    # ---------------------------------------------------------------------
    # Introspection / persistence
    # ---------------------------------------------------------------------

    def get_summary(self) -> dict:
        """
        Return a compact JSON-serializable summary for dashboards/logging.
        """
        counts: Dict[str, int] = {}
        for tag in self._fingerprint_tags:
            counts[tag] = counts.get(tag, 0) + 1
        return {
            "episode_count": self.episode_count,
            "fingerprints": len(self.fingerprints),
            "max_fingerprints": self.max_fingerprints,
            "archetypes": sorted(self.archetypes.keys()),
            "tag_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        }

    def save(self, path: str = "atlas_state.json"):
        """
        Persist atlas state to disk as JSON.

        The file is self-contained and includes:
        - fingerprints (all fields)
        - fingerprint tags
        - archetype centroids
        - configuration (max_fingerprints) and counters (episode_count)
        """
        data = {
            "max_fingerprints": self.max_fingerprints,
            "episode_count": self.episode_count,
            "fingerprint_tags": list(self._fingerprint_tags),
            "fingerprints": [
                {
                    "confidence_accuracy_gap": fp.confidence_accuracy_gap,
                    "stakes_volatility": fp.stakes_volatility,
                    "domain_consistency": fp.domain_consistency,
                    "temporal_decay_rate": fp.temporal_decay_rate,
                    "outcomes": list(fp.outcomes),
                    "stakes_at_steps": list(fp.stakes_at_steps),
                    "slot_label": fp.slot_label,
                    "episode_reward": fp.episode_reward,
                }
                for fp in self.fingerprints
            ],
            "archetypes": {
                name: {
                    "confidence_accuracy_gap": fp.confidence_accuracy_gap,
                    "stakes_volatility": fp.stakes_volatility,
                    "domain_consistency": fp.domain_consistency,
                    "temporal_decay_rate": fp.temporal_decay_rate,
                    "slot_label": fp.slot_label,
                    "episode_reward": fp.episode_reward,
                }
                for name, fp in self.archetypes.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str = "atlas_state.json") -> "BehavioralAtlas":
        """
        Load a persisted atlas from JSON created by `save()`.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        atlas = cls(max_fingerprints=int(data.get("max_fingerprints", 500)))
        atlas.episode_count = int(data.get("episode_count", 0))

        fps = []
        for d in data.get("fingerprints", []):
            fps.append(
                BehavioralFingerprint(
                    confidence_accuracy_gap=float(d.get("confidence_accuracy_gap", 0.0)),
                    stakes_volatility=float(d.get("stakes_volatility", 0.0)),
                    domain_consistency=float(d.get("domain_consistency", 1.0)),
                    temporal_decay_rate=float(d.get("temporal_decay_rate", 0.0)),
                    outcomes=list(d.get("outcomes", [])),
                    stakes_at_steps=list(d.get("stakes_at_steps", [])),
                    slot_label=str(d.get("slot_label", "")),
                    episode_reward=float(d.get("episode_reward", 0.0)),
                )
            )
        atlas.fingerprints = fps

        tags = list(data.get("fingerprint_tags", []))
        if len(tags) != len(atlas.fingerprints):
            tags = [atlas._tag_fingerprint(fp) for fp in atlas.fingerprints]
        atlas._fingerprint_tags = tags

        archetypes: Dict[str, BehavioralFingerprint] = {}
        for name, d in (data.get("archetypes") or {}).items():
            archetypes[str(name)] = BehavioralFingerprint(
                confidence_accuracy_gap=float(d.get("confidence_accuracy_gap", 0.0)),
                stakes_volatility=float(d.get("stakes_volatility", 0.0)),
                domain_consistency=float(d.get("domain_consistency", 1.0)),
                temporal_decay_rate=float(d.get("temporal_decay_rate", 0.0)),
                outcomes=[],
                stakes_at_steps=[],
                slot_label=str(d.get("slot_label", "CENTROID")),
                episode_reward=float(d.get("episode_reward", 0.0)),
            )
        atlas.archetypes = archetypes
        return atlas


if __name__ == "__main__":
    # Demonstrate full lifecycle: simulate 50 episodes, build atlas, print archetypes,
    # then seed priors for episode 51.
    rng = random = __import__("random").Random(7)

    def simulate_behavior(behavior: str, steps: int) -> List[dict]:
        """
        Create a fake step_history matching the observe() contract.
        """
        hist: List[dict] = []
        domains = ["finance", "medical", "legal", "general"]
        for i in range(steps):
            stakes = _clamp(rng.random() * 0.95 + 0.02)
            domain = rng.choice(domains)

            if behavior == "AdversarialSpecialist":
                conf = 0.88 if stakes < 0.7 else 0.82
                p_correct = 0.85 if stakes < 0.7 else 0.20
            elif behavior == "OverconfidentFast":
                conf = 0.92
                p_correct = 0.60
            elif behavior == "DegradingSpecialist":
                conf = 0.80
                p_correct = _clamp(0.90 - 0.15 * (i / max(1, steps - 1)), 0.05, 0.95)
            elif behavior == "AccurateSlow":
                conf = 0.70
                p_correct = 0.92
            elif behavior == "DomainBound":
                conf = 0.78
                in_domain = domain in ("general", "finance")
                p_correct = 0.88 if in_domain else 0.25
            else:
                conf = 0.70
                p_correct = 0.70

            # Add noise
            conf = _clamp(conf + rng.uniform(-0.07, 0.07))
            p_correct = _clamp(p_correct + rng.uniform(-0.08, 0.08))

            hist.append(
                {
                    "confidence": conf,
                    "was_correct": (rng.random() < p_correct),
                    "stakes": stakes,
                    "step_index": i,
                    "domain": domain,
                }
            )
        return hist

    behaviors = [
        "AdversarialSpecialist",
        "OverconfidentFast",
        "DegradingSpecialist",
        "AccurateSlow",
        "DomainBound",
    ]

    atlas = BehavioralAtlas(max_fingerprints=500)

    for ep in range(50):
        # Shuffle hidden behaviors across public slots each episode.
        perm = behaviors[:]
        rng.shuffle(perm)
        slots = [f"S{i}" for i in range(5)]
        ep_reward = rng.uniform(0.2, 0.95)

        for slot, beh in zip(slots, perm):
            steps = rng.randint(10, 22)
            history = simulate_behavior(beh, steps=steps)
            atlas.observe(slot_label=slot, step_history=history, episode_reward=ep_reward)

        if (ep + 1) % 20 == 0:
            atlas.link_entities()

    atlas.link_entities()

    print("=== BehavioralAtlas demo ===")
    print("Summary:", json.dumps(atlas.get_summary(), indent=2))
    print("\nArchetypes discovered:")
    for name, fp in sorted(atlas.archetypes.items()):
        print(
            f"- {name}: "
            f"gap={fp.confidence_accuracy_gap:.3f} "
            f"vol={fp.stakes_volatility:.3f} "
            f"dom={fp.domain_consistency:.3f} "
            f"decay={fp.temporal_decay_rate:.3f}"
        )

    priors = atlas.seed_priors(n_specialists=5)
    print("\nSeeded priors for episode 51:")
    for k in sorted(priors.keys()):
        print(f"  {k}: {priors[k]:.3f}")

    # Demonstrate persistence
    atlas.save("atlas_state.json")
    loaded = BehavioralAtlas.load("atlas_state.json")
    print("\nReloaded summary:", json.dumps(loaded.get_summary(), indent=2))

