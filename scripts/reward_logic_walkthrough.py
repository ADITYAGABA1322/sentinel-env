from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversary import AdversaryFSM
from audit_ledger import AuditLedger
from cluster_rewards import (
    auditor_reward,
    global_cluster_reward,
    orchestrator_reward,
    resource_manager_reward,
    task1_cluster_terminal,
    task2_cluster_terminal,
    task3_cluster_terminal,
    worker_reward,
)
from environment import SentinelEnv
from gpu_pool import GPUPool
from graders import grade_task1_step, grade_task2_step, grade_task3_step
from job_queue import GPUJob, JobQueue


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain and test SENTINEL reward logic.")
    parser.add_argument("--section", choices=["all", "current", "cluster"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.section in ("all", "current"):
        explain_current_reward_engine(args.seed)
    if args.section in ("all", "cluster"):
        explain_cluster_reward_engine(args.seed)


def explain_current_reward_engine(seed: int) -> None:
    print_rule("CURRENT REWARD ENGINE V2 - DIRECT GRADER CASES")
    cases = [
        (
            "Task1 low-stakes correct delegate",
            grade_task1_step(
                chosen_specialist="S0",
                specialist_outcome=1.0,
                stakes=0.25,
                was_adversarial=False,
                action_type="delegate",
                confidence=0.85,
                result_metadata={"step_cost": 1},
                trust_score=0.50,
            ),
            "Healthy simple step. Accuracy, stakes awareness, and efficiency are all high.",
        ),
        (
            "Task2 overconfident wrong answer",
            grade_task2_step(
                specialist_outcome=0.0,
                action_type="delegate",
                step_count=8,
                max_steps=30,
                confidence=0.95,
                result_metadata={"step_cost": 1},
            ),
            "This is the reward-hacking smell: high confidence on wrong output. Confidence alignment drops hard.",
        ),
        (
            "Task3 high-stakes poison accepted",
            grade_task3_step(
                specialist_outcome=0.0,
                stakes=0.85,
                was_adversarial=True,
                action_type="delegate",
                step_count=20,
                max_steps=45,
                confidence=0.92,
                result_metadata={"triggered": True, "threshold": 0.70},
                trust_score=0.88,
            ),
            "Bad case. The agent delegated during a high-stakes adversarial step, so task accuracy and stakes awareness collapse.",
        ),
        (
            "Task3 high-stakes adversary verified",
            grade_task3_step(
                specialist_outcome=1.0,
                stakes=0.85,
                was_adversarial=True,
                action_type="verify",
                step_count=22,
                max_steps=45,
                confidence=0.92,
                result_metadata={"triggered": True, "threshold": 0.70},
                trust_score=0.42,
            ),
            "Good case. Verification catches the adversary, so verification quality and stakes awareness become strong.",
        ),
    ]

    for title, (score, reason, breakdown), meaning in cases:
        print_case(title, score, reason, breakdown, meaning)

    print_rule("CURRENT REWARD ENGINE V2 - LIVE ENV REWARD REPORT")
    env = SentinelEnv()
    result = env.reset(task_type="task3", seed=seed)
    obs = result["observation"]
    result = env.step(
        {
            "session_id": obs["session_id"],
            "task_type": obs["task_type"],
            "action_type": "delegate",
            "specialist_id": "S0",
            "reasoning": "walkthrough first step",
        }
    )
    report = env.reward_report()
    print_json(
        {
            "step_reward": result["reward"],
            "score_so_far": result["info"]["score"],
            "reward_report": report,
        }
    )
    print(
        "\nMeaning: /reward-report is the judge-friendly audit trail. It shows every reward event, "
        "the formula components, the trust before/after, and why the score moved.\n"
    )


def explain_cluster_reward_engine(seed: int) -> None:
    print_rule("GPU CLUSTER PHASE 1 - SIMULATION INGREDIENTS")
    rng = random.Random(seed)
    pool = GPUPool(num_gpus=4, memory_per_gpu=80, failure_probability=0.0)
    queue = JobQueue(
        [
            GPUJob("JOB-001", priority=5, memory_required=48, steps_to_complete=3, deadline=8, owner="train"),
            GPUJob("JOB-002", priority=2, memory_required=24, steps_to_complete=2, deadline=7, owner="eval"),
            GPUJob("JOB-003", priority=3, memory_required=72, steps_to_complete=5, deadline=12, owner="research"),
        ]
    )

    pool.allocate("JOB-001", "GPU-00", 48)
    queue.assign("JOB-001", "GPU-00")
    pool.allocate("JOB-002", "GPU-00", 24)
    queue.assign("JOB-002", "GPU-00")
    queue.tick(current_step=1, active_job_ids={"JOB-001", "JOB-002"})

    adversary = AdversaryFSM(seed=seed, attack_probability=1.0, min_attack_gap=0)
    attack = adversary.maybe_inject(
        step=2,
        auditor_detection_rate=0.0,
        gpu_ids=["GPU-00", "GPU-01"],
        job_ids=["JOB-001", "JOB-002"],
    )

    ledger = AuditLedger()
    ledger.record_action("resource_manager", {"action_type": "allocate", "job_id": "JOB-001"}, timestamp=1)
    ledger.record_claim("resource_manager", 0.98, "claimed all jobs healthy", timestamp=2, approved=False)
    ledger.flag("resource_manager", "reward claim conflicts with job progress")

    print_json(
        {
            "gpu_pool": pool.summary(),
            "visible_gpus": pool.snapshot(include_hidden=False),
            "hidden_jobs": queue.snapshot(include_hidden=True),
            "adversary_attack": attack,
            "audit_anomaly_scores": ledger.anomaly_scores(),
        }
    )
    print(
        "\nMeaning: these are not final rewards yet. They are the raw signals the cluster reward "
        "engine will consume: utilization, deadline progress, false reports, attacks, and audit anomalies.\n"
    )

    print_rule("GPU CLUSTER REWARD FORMULAS - PER AGENT")
    orchestrator, orchestrator_breakdown = orchestrator_reward(
        goal_completion_rate=queue.completion_rate(),
        plan_coherence_score=0.72,
        recovery_speed=0.60,
    )
    resource_mgr, resource_breakdown = resource_manager_reward(
        gpu_utilization_efficiency=pool.utilization_rate(),
        deadline_hit_rate=queue.deadline_hit_rate(),
        waste_penalty=0.18,
    )
    auditor, auditor_breakdown = auditor_reward(
        detection_rate=0.75,
        false_positive_rate=0.10,
        calibration_score=0.70,
    )
    worker, worker_breakdown = worker_reward(
        job_completion_accuracy=0.66,
        report_honesty_score=0.55,
    )
    global_score, global_breakdown = global_cluster_reward(
        {
            "orchestrator": orchestrator,
            "resource_manager": resource_mgr,
            "auditor": auditor,
            "worker": worker,
        },
        cluster_health_score=pool.cluster_health_score(),
    )
    print_json(
        {
            "orchestrator": {"reward": orchestrator, "breakdown": orchestrator_breakdown},
            "resource_manager": {"reward": resource_mgr, "breakdown": resource_breakdown},
            "auditor": {"reward": auditor, "breakdown": auditor_breakdown},
            "worker": {"reward": worker, "breakdown": worker_breakdown},
            "global_cluster_reward": {"reward": global_score, "breakdown": global_breakdown},
        }
    )
    print(
        "\nMeaning: per-agent rewards can be decent, but the global reward is multiplied by cluster health. "
        "That is the anti-hack design: individual gaming cannot win if the cluster degrades.\n"
    )

    print_rule("GPU CLUSTER TASK TERMINAL REWARDS")
    task1 = task1_cluster_terminal(jobs_completed_rate=0.82, avg_gpu_utilization=0.74)
    task2 = task2_cluster_terminal(
        jobs_completed_rate=0.76,
        worker_trust_calibration=0.68,
        deadline_recovery_rate=0.61,
    )
    task3 = task3_cluster_terminal(
        jobs_completed_rate=0.70,
        adversarial_detection_rate=0.80,
        reward_hack_detection_rate=0.75,
        plan_coherence_score=0.66,
        efficiency_score=0.58,
    )
    print_json(
        {
            "task1_cluster_basics": {"reward": task1[0], "breakdown": task1[1]},
            "task2_unreliable_workers": {"reward": task2[0], "breakdown": task2[1]},
            "task3_full_adversarial_cluster": {"reward": task3[0], "breakdown": task3[1]},
        }
    )
    print(
        "\nMeaning: these are the terminal scores for the GPU-cluster version. "
        "Task3 is intentionally multi-objective: complete jobs, catch adversary, catch reward hacks, keep plan coherence, stay efficient.\n"
    )


def print_case(title: str, score: float, reason: str, breakdown: dict[str, Any], meaning: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print_json({"reward": round(score, 4), "reason": reason, "breakdown": breakdown})
    print(f"Meaning: {meaning}")


def print_rule(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
