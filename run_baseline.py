"""
SENTINEL — Baseline Runner
=========================

Runs two baseline strategies (random, heuristic) for 100 episodes each across
task1/task2/task3 against the FastAPI environment.

Usage:
    python run_baseline.py

Env vars (optional):
    ENV_URL    Base URL for the env (default: http://127.0.0.1:7860)
    TIMEOUT_S  HTTP timeout seconds (default: 30)
"""

from __future__ import annotations

import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx


TaskType = Literal["task1", "task2", "task3"]
Strategy = Literal["random", "heuristic"]
ActionType = Literal["delegate", "verify", "solve_independently", "skip"]

TASKS: list[TaskType] = ["task1", "task2", "task3"]
STRATEGIES: list[Strategy] = ["random", "heuristic"]
SPECIALISTS: list[str] = ["S0", "S1", "S2", "S3", "S4"]
ACTION_TYPES: list[ActionType] = ["delegate", "verify", "solve_independently", "skip"]


@dataclass(frozen=True)
class EpisodeRecord:
    task_type: TaskType
    strategy: Strategy
    episode_id: str
    episode_index: int
    total_reward: float
    steps_taken: int
    adversarial_detections: int
    total_adversarial_encounters: int
    subtasks_completed: int


class EnvClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)
        self.session_id: str = ""

    def health(self) -> dict[str, Any]:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def reset(self, task_type: TaskType, seed: int | None = None, scenario_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"task_type": task_type, "seed": seed, "scenario_id": scenario_id}
        r = self._client.post("/reset", json=payload)
        r.raise_for_status()
        result = r.json()
        self.session_id = result["info"]["session_id"]
        return result

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        action = dict(action)
        action["session_id"] = self.session_id
        r = self._client.post(f"/step?session_id={self.session_id}", json=action)
        r.raise_for_status()
        return r.json()


def _best_specialist_by_trust(trust: dict[str, float], available: list[str]) -> tuple[str, float]:
    def key(s: str) -> tuple[float, int, str]:
        # Prefer higher trust; if equal, prefer S0; then stable by id.
        return (float(trust.get(s, 0.5)), 1 if s == "S0" else 0, s)

    best = sorted(available, key=key, reverse=True)[0]
    return best, float(trust.get(best, 0.5))


def choose_action(
    *,
    strategy: Strategy,
    obs: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    task_type: TaskType = obs["task_type"]

    if strategy == "random":
        action_type: ActionType = rng.choice(ACTION_TYPES)
        specialist_id: str | None = None
        subtask_response: str | None = None

        if action_type in ("delegate", "verify"):
            specialist_id = rng.choice(SPECIALISTS)
        elif action_type == "solve_independently":
            subtask_response = "SOLVED"

        return {
            "task_type": task_type,
            "action_type": action_type,
            "specialist_id": specialist_id,
            "subtask_response": subtask_response,
            "reasoning": f"random(seed-step)",
        }

    # heuristic
    trust: dict[str, float] = obs["trust_snapshot"]
    stakes: float = float(obs["stakes_level"])
    available: list[str] = list(obs["available_specialists"])

    best, best_trust = _best_specialist_by_trust(trust, available)

    # Spec-required rule:
    # verify if stakes_level > 0.7 and top specialist trust < 0.6
    action_type = "verify" if (stakes > 0.7 and best_trust < 0.6) else "delegate"

    return {
        "task_type": task_type,
        "action_type": action_type,
        "specialist_id": best,
        "subtask_response": None,
        "reasoning": f"trust={best}:{best_trust:.3f} stakes={stakes:.2f}",
    }


def _count_verify_or_selfsolve_adversarial_detections(reward_report: dict[str, Any]) -> int:
    events = reward_report.get("events") or []
    detections = 0
    for ev in events:
        if ev.get("kind") != "step":
            continue
        if ev.get("action_type") not in ("verify", "solve_independently"):
            continue
        if bool(ev.get("was_adversarial")):
            detections += 1
    return detections


def run_episode(
    *,
    client: EnvClient,
    task_type: TaskType,
    strategy: Strategy,
    episode_index: int,
    seed: int,
) -> EpisodeRecord:
    rng = random.Random(seed)
    scenario_label = f"SCN-{task_type.upper()}-{strategy.upper()}-{episode_index+1:03d}"

    result = client.reset(task_type=task_type, seed=seed, scenario_id=None)
    episode_id = str(result["info"]["episode_id"])

    print(f"[START] task={scenario_label} env=sentinel-env model={strategy}-baseline")

    step_num = 0
    rewards: list[float] = []
    final_score = 0.0

    while True:
        obs = result["observation"]
        action = choose_action(strategy=strategy, obs=obs, rng=rng)

        try:
            result = client.step(action)
        except Exception as e:
            print(f"[STEP] step=0 action=error reward=0.00 done=true error={e}")
            print("[END] success=false steps=0 score=0.000 rewards=0.00")
            raise

        reward = float(result["reward"]["value"])
        done = bool(result["done"])
        step_num += 1
        rewards.append(reward)
        final_score = float(result["info"].get("score", 0.0))

        action_str = f"{action['action_type']}:{action.get('specialist_id','SELF')}"
        print(
            f"[STEP] step={step_num} "
            f"action={action_str} "
            f"reward={reward:.2f} "
            f"done={str(done).lower()} "
            f"error=null"
        )

        if done:
            break

    info = result["info"]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success=true "
        f"steps={step_num} "
        f"score={final_score:.3f} "
        f"rewards={rewards_str}"
    )

    reward_report = info.get("reward_report") or {}
    adversarial_detections = _count_verify_or_selfsolve_adversarial_detections(reward_report)

    # "Encounters" = detections + poisonings (matches env's own adaptive stats definition).
    poisonings = int(info.get("adversarial_poisonings", 0) or 0)
    detections_total = int(info.get("adversarial_detections", 0) or 0)
    total_adversarial_encounters = int(detections_total + poisonings)

    return EpisodeRecord(
        task_type=task_type,
        strategy=strategy,
        episode_id=episode_id,
        episode_index=episode_index,
        total_reward=float(info.get("total_reward", 0.0) or 0.0),
        steps_taken=step_num,
        adversarial_detections=int(adversarial_detections),
        total_adversarial_encounters=total_adversarial_encounters,
        subtasks_completed=int(info.get("subtasks_completed", 0) or 0),
    )


def _fmt_table(rows: list[list[str]], headers: list[str]) -> str:
    table = [headers] + rows
    widths = [max(len(str(r[i])) for r in table) for i in range(len(headers))]
    lines = []
    lines.append("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("  ".join(("-" * widths[i]) for i in range(len(headers))))
    for r in rows:
        lines.append("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def main() -> None:
    base_url = os.environ.get("ENV_URL", "http://127.0.0.1:7860").strip()
    timeout_s = float(os.environ.get("TIMEOUT_S", "30").strip() or "30")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = EnvClient(base_url=base_url, timeout_s=timeout_s)
        client.health()
    except Exception as e:
        print(f"Could not connect to SENTINEL env at {base_url}.")
        print(f"Error: {e}")
        print("Make sure the FastAPI server is running on port 7860, then retry.")
        sys.exit(2)

    records: list[EpisodeRecord] = []

    for task_type in TASKS:
        for strategy in STRATEGIES:
            for i in range(100):
                seed = (hash((task_type, strategy)) & 0xFFFF) + i
                try:
                    rec = run_episode(
                        client=client,
                        task_type=task_type,
                        strategy=strategy,
                        episode_index=i,
                        seed=seed,
                    )
                    records.append(rec)
                except Exception:
                    # keep going; error already printed in episode logs
                    continue

    # Save CSV
    csv_path = outputs_dir / "baseline_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_type",
                "strategy",
                "episode_id",
                "episode_index",
                "total_reward",
                "steps_taken",
                "adversarial_detections",
                "total_adversarial_encounters",
                "subtasks_completed",
            ],
        )
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "task_type": r.task_type,
                    "strategy": r.strategy,
                    "episode_id": r.episode_id,
                    "episode_index": r.episode_index,
                    "total_reward": f"{r.total_reward:.4f}",
                    "steps_taken": r.steps_taken,
                    "adversarial_detections": r.adversarial_detections,
                    "total_adversarial_encounters": r.total_adversarial_encounters,
                    "subtasks_completed": r.subtasks_completed,
                }
            )

    # Summary stats
    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    summary_rows: list[list[str]] = []
    mean_reward: dict[tuple[TaskType, Strategy], float] = {}
    detection_rate: dict[tuple[TaskType, Strategy], float] = {}

    for task_type in TASKS:
        for strategy in STRATEGIES:
            rs = [r for r in records if r.task_type == task_type and r.strategy == strategy]
            rewards = [r.total_reward for r in rs]
            steps = [float(r.steps_taken) for r in rs]
            det = sum(r.adversarial_detections for r in rs)
            enc = sum(r.total_adversarial_encounters for r in rs)
            rate = (det / enc) if enc > 0 else 0.0

            mr = mean(rewards)
            mean_reward[(task_type, strategy)] = mr
            detection_rate[(task_type, strategy)] = rate

            summary_rows.append(
                [
                    task_type,
                    strategy,
                    str(len(rs)),
                    f"{mr:.4f}",
                    f"{mean(steps):.2f}",
                    f"{rate:.3f}",
                ]
            )

    print()
    print(
        _fmt_table(
            summary_rows,
            headers=["task", "strategy", "episodes", "mean_total_reward", "mean_steps", "detection_rate"],
        )
    )
    print()
    print(f"Saved CSV: {csv_path}")

    # Plots (matplotlib)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Matplotlib not available, skipping plots. Error: {e}")
        return

    # Plot 1: mean reward grouped bars by task
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(TASKS)))
    width = 0.35
    random_vals = [mean_reward.get((t, "random"), 0.0) for t in TASKS]
    heuristic_vals = [mean_reward.get((t, "heuristic"), 0.0) for t in TASKS]
    ax1.bar([i - width / 2 for i in x], random_vals, width, label="random")
    ax1.bar([i + width / 2 for i in x], heuristic_vals, width, label="heuristic")
    ax1.set_xticks(x, TASKS)
    ax1.set_ylabel("Mean total reward")
    ax1.set_title("Mean reward by task and strategy")
    ax1.legend()
    fig1.tight_layout()
    out1 = outputs_dir / "reward_by_task_and_strategy.png"
    fig1.savefig(out1, dpi=160)
    plt.close(fig1)

    # Plot 2: detection rate per task, both strategies
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    random_rates = [detection_rate.get((t, "random"), 0.0) for t in TASKS]
    heuristic_rates = [detection_rate.get((t, "heuristic"), 0.0) for t in TASKS]
    ax2.bar([i - width / 2 for i in x], random_rates, width, label="random")
    ax2.bar([i + width / 2 for i in x], heuristic_rates, width, label="heuristic")
    ax2.set_xticks(x, TASKS)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Detection rate (detections / encounters)")
    ax2.set_title("Detection rate by task (both strategies)")
    ax2.legend()
    fig2.tight_layout()
    out2 = outputs_dir / "detection_rate_by_task.png"
    fig2.savefig(out2, dpi=160)
    plt.close(fig2)

    print(f"Saved plot: {out1}")
    print(f"Saved plot: {out2}")


if __name__ == "__main__":
    main()

