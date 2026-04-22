from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import SentinelEnv, _GROUND_TRUTH_RELIABILITY


Policy = Callable[[SentinelEnv, dict, random.Random], dict]


def random_policy(env: SentinelEnv, obs: dict, rng: random.Random) -> dict:
    specialist = rng.choice(obs["available_specialists"])
    roll = rng.random()
    if roll < 0.65:
        action_type = "delegate"
    elif roll < 0.85:
        action_type = "verify"
    elif roll < 0.95:
        action_type = "solve_independently"
        specialist = None
    else:
        action_type = "skip"
        specialist = None
    return _action(obs, action_type, specialist)


def heuristic_policy(env: SentinelEnv, obs: dict, rng: random.Random) -> dict:
    trust = obs["trust_snapshot"]
    specialist = max(obs["available_specialists"], key=lambda sid: trust.get(sid, 0.5))
    action_type = "verify" if obs["stakes_level"] >= 0.70 and trust.get(specialist, 0.5) < 0.65 else "delegate"
    return _action(obs, action_type, specialist)


def oracle_lite_policy(env: SentinelEnv, obs: dict, rng: random.Random) -> dict:
    reliability = env._pool.public_ground_truth_reliability(_GROUND_TRUTH_RELIABILITY)
    if obs["task_type"] == "task3" and obs["stakes_level"] >= 0.70:
        return _action(obs, "verify", env._pool.adversarial_slot)
    specialist = max(obs["available_specialists"], key=lambda sid: reliability.get(sid, 0.5))
    return _action(obs, "delegate", specialist)


def _action(obs: dict, action_type: str, specialist_id: str | None) -> dict:
    return {
        "session_id": obs["session_id"],
        "task_type": obs["task_type"],
        "action_type": action_type,
        "specialist_id": specialist_id,
        "subtask_response": "SELF_SOLVED" if action_type == "solve_independently" else None,
        "reasoning": f"{action_type} via {specialist_id or 'SELF'}",
    }


def run_episode(policy_name: str, policy: Policy, task_type: str, seed: int) -> dict:
    rng = random.Random(seed)
    env = SentinelEnv()
    result = env.reset(task_type=task_type, seed=seed)
    rewards: list[float] = []

    while not result["done"]:
        action = policy(env, result["observation"], rng)
        result = env.step(action)
        rewards.append(result["reward"]["value"])

    info = result["info"]
    breakdown = result["reward"]["signal_breakdown"]
    detections = info.get("adversarial_detections", 0)
    poisonings = info.get("adversarial_poisonings", 0)
    total_adversarial = detections + poisonings
    detection_rate = detections / total_adversarial if total_adversarial else breakdown.get("detection_rate", 1.0)

    return {
        "policy": policy_name,
        "task_type": task_type,
        "seed": seed,
        "steps": info.get("step_count", 0),
        "score": round(info.get("score", 0.0), 4),
        "total_reward": round(info.get("total_reward", 0.0), 4),
        "completion_rate": round(info.get("completion_rate", 0.0), 4),
        "detection_rate": round(detection_rate, 4),
        "trust_calibration": round(breakdown.get("trust_calibration", 0.0), 4),
        "adversarial_detections": detections,
        "adversarial_poisonings": poisonings,
        "status": "failed" if info.get("forced_end") else "completed",
        "rewards": [round(value, 4) for value in rewards],
    }


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["policy"], []).append(row)

    summary = {}
    for policy_name, policy_rows in grouped.items():
        n = len(policy_rows)
        summary[policy_name] = {
            "episodes": n,
            "avg_score": _avg(policy_rows, "score"),
            "avg_completion_rate": _avg(policy_rows, "completion_rate"),
            "avg_detection_rate": _avg(policy_rows, "detection_rate"),
            "avg_trust_calibration": _avg(policy_rows, "trust_calibration"),
            "avg_steps": _avg(policy_rows, "steps"),
        }
    return summary


def _avg(rows: list[dict], key: str) -> float:
    return round(sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows)), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SENTINEL policies.")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per policy.")
    parser.add_argument("--task", default="task3", choices=["task1", "task2", "task3"])
    parser.add_argument("--out", default="outputs/evaluation_results.json")
    args = parser.parse_args()

    policies: dict[str, Policy] = {
        "random": random_policy,
        "heuristic": heuristic_policy,
        "oracle_lite": oracle_lite_policy,
    }

    rows = []
    for policy_name, policy in policies.items():
        for seed in range(args.episodes):
            rows.append(run_episode(policy_name, policy, args.task, seed))

    payload = {
        "task": args.task,
        "episodes_per_policy": args.episodes,
        "summary": summarize(rows),
        "episodes": rows,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
