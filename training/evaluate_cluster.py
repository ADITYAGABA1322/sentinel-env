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

from cluster_trust_env import ClusterTrustEnv
from difficulty_controller import GLOBAL_DIFFICULTY_CONTROLLER


Policy = Callable[[ClusterTrustEnv, dict, random.Random], dict]


GROUND_TRUTH_RELIABILITY = {
    "reliable": 0.93,
    "slow": 0.78,
    "degrading": 0.62,
    "overreporter": 0.45,
    "adversarial": 0.18,
}


def random_policy(env: ClusterTrustEnv, obs: dict, rng: random.Random) -> dict:
    action_type = rng.choice(["allocate", "allocate", "request_info", "verify", "preempt", "tick"])
    return _action_for_type(obs, action_type, rng, worker_id=rng.choice(obs["available_workers"]))


def blind_policy(env: ClusterTrustEnv, obs: dict, rng: random.Random) -> dict:
    return _action_for_type(obs, "allocate", rng, worker_id=obs["available_workers"][0])


def trust_policy(env: ClusterTrustEnv, obs: dict, rng: random.Random) -> dict:
    running = _jobs_with_status(obs, "running")
    anomaly = max(obs["audit_anomaly_scores"].values() or [0.0])
    if running and (obs["step_count"] % 4 == 0 or anomaly > 0.55):
        return _action_for_type(obs, "verify", rng)
    if _jobs_with_status(obs, "queued"):
        return _action_for_type(obs, "allocate", rng, worker_id=_trust_calibrated_worker(obs))
    if running:
        return _action_for_type(obs, "request_info", rng)
    return {"session_id": obs["session_id"], "action_type": "tick"}


def oracle_lite_policy(env: ClusterTrustEnv, obs: dict, rng: random.Random) -> dict:
    running = _jobs_with_status(obs, "running")
    state = env.state()
    profile = state["worker_profile_hidden"]
    best_worker = max(profile, key=lambda sid: GROUND_TRUTH_RELIABILITY[profile[sid]])
    if running and state["attack_attempts"] > state["attack_detections"]:
        return _action_for_type(obs, "verify", rng)
    if _jobs_with_status(obs, "queued"):
        return _action_for_type(obs, "allocate", rng, worker_id=best_worker)
    if running:
        return _action_for_type(obs, "request_info", rng, worker_id=best_worker)
    return {"session_id": obs["session_id"], "action_type": "tick"}


def _action_for_type(obs: dict, action_type: str, rng: random.Random, worker_id: str | None = None) -> dict:
    session_id = obs["session_id"]
    running = _jobs_with_status(obs, "running")
    pending = _jobs_with_status(obs, "queued")

    if action_type == "allocate" and pending:
        job = min(pending, key=lambda row: (row["deadline"], -row["memory_required"]))
        gpu = max(obs["gpu_pool"], key=lambda row: row["memory_free"])
        return {
            "session_id": session_id,
            "action_type": "allocate",
            "job_id": job["job_id"],
            "gpu_id": gpu["id"],
            "worker_id": worker_id or _trust_calibrated_worker(obs),
        }
    if action_type in {"verify", "request_info", "preempt"} and running:
        job = min(running, key=lambda row: row["deadline"])
        payload = {"session_id": session_id, "action_type": action_type, "job_id": job["job_id"]}
        if worker_id:
            payload["worker_id"] = worker_id
        return payload
    return {"session_id": session_id, "action_type": "tick"}


def _jobs_with_status(obs: dict, status: str) -> list[dict]:
    return [job for job in obs["job_queue_summary"] if job["status"] == status]


def _trust_calibrated_worker(obs: dict) -> str:
    fingerprints = obs["behavioral_fingerprints"]
    trust = obs["trust_snapshot"]

    def score(worker_id: str) -> float:
        calls = fingerprints[worker_id]["calls"]
        gap = fingerprints[worker_id]["confidence_accuracy_gap"]
        return trust[worker_id] + 0.25 / (1 + calls) - 0.20 * gap

    min_calls = min(fingerprints[worker_id]["calls"] for worker_id in obs["available_workers"])
    if min_calls < 2:
        under_observed = [
            worker_id for worker_id in obs["available_workers"]
            if fingerprints[worker_id]["calls"] == min_calls
        ]
        return max(under_observed, key=score)
    return max(obs["available_workers"], key=score)


def run_episode(policy_name: str, policy: Policy, task_type: str, seed: int, adaptive: bool = False) -> dict:
    rng = random.Random(seed)
    env = ClusterTrustEnv()
    result = env.reset(task_type=task_type, seed=seed, adaptive=adaptive)
    rewards: list[float] = []

    while not result["done"]:
        action = policy(env, result["observation"], rng)
        result = env.step(action)
        rewards.append(result["reward"]["value"])

    state = env.state()
    coverage = state["ai_failure_coverage"]
    attacks = state["attack_detections"] + state["attack_poisonings"]
    detection_rate = state["attack_detections"] / max(1, attacks)

    return {
        "policy": policy_name,
        "task_type": task_type,
        "seed": seed,
        "steps": state["step_count"],
        "score": round(state["score"], 4),
        "cluster_health": state["cluster"]["cluster_health_score"],
        "utilization_rate": state["cluster"]["utilization_rate"],
        "completion_rate": state["jobs"]["completion_rate"],
        "deadline_hit_rate": state["jobs"]["deadline_hit_rate"],
        "detection_rate": round(detection_rate, 4),
        "attack_detections": state["attack_detections"],
        "attack_poisonings": state["attack_poisonings"],
        "ai_reliability_modifier": coverage["ai_reliability_modifier"],
        "context_drift_events": coverage["context_memory_loss"]["drift_events"],
        "loop_events": coverage["agent_loop_reliability"]["loop_events"],
        "hallucination_confidence_score": coverage["hallucination_confidence"]["score"],
        "evaluation_freshness_score": coverage["evaluation_collapse"]["score"],
        "trust_snapshot": state["trust_snapshot"],
        "difficulty_profile": state["difficulty_profile"],
        "rewards": [round(value, 4) for value in rewards],
    }


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["policy"], []).append(row)

    return {
        policy: {
            "episodes": len(items),
            "avg_score": _avg(items, "score"),
            "avg_cluster_health": _avg(items, "cluster_health"),
            "avg_utilization_rate": _avg(items, "utilization_rate"),
            "avg_completion_rate": _avg(items, "completion_rate"),
            "avg_detection_rate": _avg(items, "detection_rate"),
            "avg_ai_reliability_modifier": _avg(items, "ai_reliability_modifier"),
            "avg_steps": _avg(items, "steps"),
        }
        for policy, items in sorted(grouped.items())
    }


def _avg(rows: list[dict], key: str) -> float:
    return round(sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows)), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SENTINEL GPU-cluster policies.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--task", default="task3", choices=["task1", "task2", "task3", "all"])
    parser.add_argument("--out", default="outputs/cluster_evaluation_results.json")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--reset-difficulty", action="store_true")
    args = parser.parse_args()

    if args.reset_difficulty:
        GLOBAL_DIFFICULTY_CONTROLLER.reset()

    policies: dict[str, Policy] = {
        "random": random_policy,
        "blind": blind_policy,
        "trust": trust_policy,
        "oracle_lite": oracle_lite_policy,
    }
    tasks = ["task1", "task2", "task3"] if args.task == "all" else [args.task]
    rows = [
        run_episode(policy_name, policy, task_type, seed, adaptive=args.adaptive)
        for task_type in tasks
        for policy_name, policy in policies.items()
        for seed in range(args.episodes)
    ]
    payload = {
        "environment": "cluster",
        "tasks": tasks,
        "episodes_per_policy": args.episodes,
        "adaptive": args.adaptive,
        "difficulty_controller": GLOBAL_DIFFICULTY_CONTROLLER.state(),
        "summary": summarize(rows),
        "episodes": rows,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps({"summary": payload["summary"], "out": str(out_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
