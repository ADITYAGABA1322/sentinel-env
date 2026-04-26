from __future__ import annotations

from typing import Mapping


def clamp_reward(value: float) -> float:
    """Boundary-exclusive reward in the OpenEnv convention."""
    return round(max(0.01, min(0.99, value)), 4)


def orchestrator_reward(
    goal_completion_rate: float,
    plan_coherence_score: float,
    recovery_speed: float,
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "goal_completion_rate": _unit(goal_completion_rate),
        "plan_coherence_score": _unit(plan_coherence_score),
        "recovery_speed": _unit(recovery_speed),
    }
    score = (
        0.40 * breakdown["goal_completion_rate"]
      + 0.30 * breakdown["plan_coherence_score"]
      + 0.30 * breakdown["recovery_speed"]
    )
    return clamp_reward(score), breakdown


def resource_manager_reward(
    gpu_utilization_efficiency: float,
    deadline_hit_rate: float,
    waste_penalty: float,
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "gpu_utilization_efficiency": _unit(gpu_utilization_efficiency),
        "deadline_hit_rate": _unit(deadline_hit_rate),
        "waste_penalty": _unit(waste_penalty),
    }
    score = (
        0.50 * breakdown["gpu_utilization_efficiency"]
      + 0.30 * breakdown["deadline_hit_rate"]
      - 0.20 * breakdown["waste_penalty"]
    )
    return clamp_reward(score), breakdown


def auditor_reward(
    detection_rate: float,
    false_positive_rate: float,
    calibration_score: float,
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "detection_rate": _unit(detection_rate),
        "false_positive_rate": _unit(false_positive_rate),
        "calibration_score": _unit(calibration_score),
    }
    score = (
        0.50 * breakdown["detection_rate"]
      - 0.30 * breakdown["false_positive_rate"]
      + 0.20 * breakdown["calibration_score"]
    )
    return clamp_reward(score), breakdown


def worker_reward(
    job_completion_accuracy: float,
    report_honesty_score: float,
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "job_completion_accuracy": _unit(job_completion_accuracy),
        "report_honesty_score": _unit(report_honesty_score),
    }
    score = (
        0.70 * breakdown["job_completion_accuracy"]
      + 0.30 * breakdown["report_honesty_score"]
    )
    return clamp_reward(score), breakdown


def adversary_reward(
    successful_disruptions: float,
    detection_penalty: float,
    curriculum_bonus: float,
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "successful_disruptions": _unit(successful_disruptions),
        "detection_penalty": _unit(detection_penalty),
        "curriculum_bonus": _unit(curriculum_bonus),
    }
    score = (
        0.60 * breakdown["successful_disruptions"]
      - 0.40 * breakdown["detection_penalty"]
      + 0.10 * breakdown["curriculum_bonus"]
    )
    return clamp_reward(score), breakdown


def global_cluster_reward(
    agent_rewards: Mapping[str, float],
    cluster_health_score: float,
    reliability_modifier: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """
    Collective reward. Any cluster collapse multiplies the useful agent work down.

    The adversary is intentionally excluded from global defender reward.
    """
    weighted = (
        0.30 * agent_rewards.get("orchestrator", 0.0)
      + 0.30 * agent_rewards.get("resource_manager", 0.0)
      + 0.20 * agent_rewards.get("auditor", 0.0)
      + 0.20 * agent_rewards.get("worker", 0.0)
    )
    health = _unit(cluster_health_score)
    reliability = _unit(reliability_modifier)
    score = weighted * health * reliability
    return clamp_reward(score), {
        "weighted_agent_score": round(weighted, 4),
        "cluster_health_score": health,
        "ai_reliability_modifier": reliability,
        "orchestrator": round(agent_rewards.get("orchestrator", 0.0), 4),
        "resource_manager": round(agent_rewards.get("resource_manager", 0.0), 4),
        "auditor": round(agent_rewards.get("auditor", 0.0), 4),
        "worker": round(agent_rewards.get("worker", 0.0), 4),
    }


def ai_reliability_modifier(
    loop_avoidance: float,
    context_memory_score: float,
    hallucination_resistance: float,
    evaluation_freshness: float,
) -> tuple[float, dict[str, float]]:
    """
    Cross-cutting real-world AI reliability score.

    This turns common agent failure modes into an explicit reward multiplier.
    It does not replace task reward; it prevents brittle agents from scoring
    well while looping, drifting, trusting confident lies, or memorizing evals.
    """
    breakdown = {
        "loop_avoidance": _unit(loop_avoidance),
        "context_memory_score": _unit(context_memory_score),
        "hallucination_resistance": _unit(hallucination_resistance),
        "evaluation_freshness": _unit(evaluation_freshness),
    }
    score = (
        0.30 * breakdown["loop_avoidance"]
      + 0.30 * breakdown["context_memory_score"]
      + 0.25 * breakdown["hallucination_resistance"]
      + 0.15 * breakdown["evaluation_freshness"]
    )
    return _unit(score), breakdown


def task1_cluster_terminal(
    jobs_completed_rate: float,
    avg_gpu_utilization: float,
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "jobs_completed_rate": _unit(jobs_completed_rate),
        "avg_gpu_utilization": _unit(avg_gpu_utilization),
    }
    score = (
        0.60 * breakdown["jobs_completed_rate"]
      + 0.40 * breakdown["avg_gpu_utilization"]
    )
    return clamp_reward(score), breakdown


def task2_cluster_terminal(
    jobs_completed_rate: float,
    worker_trust_calibration: float,
    deadline_recovery_rate: float,
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "jobs_completed_rate": _unit(jobs_completed_rate),
        "worker_trust_calibration": _unit(worker_trust_calibration),
        "deadline_recovery_rate": _unit(deadline_recovery_rate),
    }
    score = (
        0.40 * breakdown["jobs_completed_rate"]
      + 0.30 * breakdown["worker_trust_calibration"]
      + 0.30 * breakdown["deadline_recovery_rate"]
    )
    return clamp_reward(score), breakdown


def task3_cluster_terminal(
    jobs_completed_rate: float,
    adversarial_detection_rate: float,
    reward_hack_detection_rate: float,
    plan_coherence_score: float,
    efficiency_score: float,
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "jobs_completed_rate": _unit(jobs_completed_rate),
        "adversarial_detection_rate": _unit(adversarial_detection_rate),
        "reward_hack_detection_rate": _unit(reward_hack_detection_rate),
        "plan_coherence_score": _unit(plan_coherence_score),
        "efficiency_score": _unit(efficiency_score),
    }
    score = (
        0.30 * breakdown["jobs_completed_rate"]
      + 0.25 * breakdown["adversarial_detection_rate"]
      + 0.20 * breakdown["reward_hack_detection_rate"]
      + 0.15 * breakdown["plan_coherence_score"]
      + 0.10 * breakdown["efficiency_score"]
    )
    return clamp_reward(score), breakdown


def _unit(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)
