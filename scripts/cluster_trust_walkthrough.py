from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster_trust_env import ClusterTrustEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the combined GPU + trust SENTINEL environment.")
    parser.add_argument("--task", choices=["task1", "task2", "task3"], default="task3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--policy", choices=["trust", "blind"], default="trust")
    args = parser.parse_args()

    env = ClusterTrustEnv()
    result = env.reset(task_type=args.task, seed=args.seed)
    rng = random.Random(args.seed)

    print("=" * 100)
    print("SENTINEL COMBINED GPU + TRUST WALKTHROUGH")
    print("=" * 100)
    print(f"task={args.task} seed={args.seed} policy={args.policy}")
    print()
    print("RESET OBSERVATION - compact")
    print(json.dumps(compact_obs(result["observation"]), indent=2))
    print()
    print("HIDDEN WORKER PROFILE - builder only")
    print(json.dumps(env.state()["worker_profile_hidden"], indent=2))
    print()
    print("step | action            | reward | score | health | util  | ai-rel | jobs done | attacks det/pois | trust")
    print("-" * 132)

    for _ in range(args.steps):
        if result["done"]:
            break
        obs = result["observation"]
        action = choose_action(obs, args.policy, rng)
        result = env.step(action)
        state = env.state()
        trust = " ".join(f"{k}:{v:.2f}" for k, v in state["trust_snapshot"].items())
        print(
            f"{state['step_count']:>4} | {action['action_type'] + ':' + str(action.get('worker_id') or action.get('job_id') or ''):<17} "
            f"| {result['reward']['value']:<6.3f} | {state['score']:<5.3f} | "
            f"{state['cluster']['cluster_health_score']:<6.2f} | {state['cluster']['utilization_rate']:<5.2f} | "
            f"{state['ai_failure_coverage']['ai_reliability_modifier']:<6.2f} | "
            f"{state['jobs']['statuses']['complete']:>3}/{state['jobs']['jobs_total']:<3} | "
            f"{state['attack_detections']:>3}/{state['attack_poisonings']:<3} | {trust}"
        )
        print(f"     reason: {result['reward']['reason']}")

    print()
    print("FINAL STATE")
    print(json.dumps(env.state(), indent=2))
    print()
    print("REWARD REPORT - last 3 events")
    report = env.reward_report()
    report["events"] = report["events"][-3:]
    print(json.dumps(report, indent=2))


def choose_action(obs: dict, policy: str, rng: random.Random) -> dict:
    session_id = obs["session_id"]
    running = [job for job in obs["job_queue_summary"] if job["status"] == "running"]
    pending = [job for job in obs["job_queue_summary"] if job["status"] == "queued"]
    trust = obs["trust_snapshot"]

    if running and (obs["step_count"] % 4 == 0 or max(obs["audit_anomaly_scores"].values() or [0.0]) > 0.6):
        job = min(running, key=lambda row: row["deadline"])
        return {
            "session_id": session_id,
            "action_type": "verify",
            "job_id": job["job_id"],
        }

    if pending:
        job = min(pending, key=lambda row: row["deadline"])
        gpu = max(obs["gpu_pool"], key=lambda row: row["memory_free"])
        worker_id = select_worker(obs) if policy == "trust" else obs["available_workers"][0]
        return {
            "session_id": session_id,
            "action_type": "allocate",
            "job_id": job["job_id"],
            "gpu_id": gpu["id"],
            "worker_id": worker_id,
        }

    if running:
        job = rng.choice(running)
        return {"session_id": session_id, "action_type": "request_info", "job_id": job["job_id"]}

    return {"session_id": session_id, "action_type": "tick"}


def select_worker(obs: dict) -> str:
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


def compact_obs(obs: dict) -> dict:
    return {
        "session_id": obs["session_id"],
        "task_type": obs["task_type"],
        "step_count": obs["step_count"],
        "max_steps": obs["max_steps"],
        "cluster_health": obs["cluster_health"],
        "utilization_rate": obs["utilization_rate"],
        "pending_jobs": sum(1 for job in obs["job_queue_summary"] if job["status"] == "queued"),
        "running_jobs": sum(1 for job in obs["job_queue_summary"] if job["status"] == "running"),
        "trust_snapshot": obs["trust_snapshot"],
        "audit_anomaly_scores": obs["audit_anomaly_scores"],
        "ai_failure_coverage": {
            "agent_loop_reliability": obs["ai_failure_coverage"]["agent_loop_reliability"],
            "context_memory_loss": obs["ai_failure_coverage"]["context_memory_loss"],
            "hallucination_confidence": obs["ai_failure_coverage"]["hallucination_confidence"],
            "evaluation_collapse": obs["ai_failure_coverage"]["evaluation_collapse"],
        },
        "allowed_actions": obs["allowed_actions"],
    }


if __name__ == "__main__":
    main()
