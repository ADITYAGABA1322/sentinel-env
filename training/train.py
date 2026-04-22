from __future__ import annotations

"""
Minimal onsite training entrypoint.

This file is intentionally import-light so it can run locally without GPU
packages. On the finale machine, install the training extras from pyproject and
use this script as the GRPO wiring point.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import SentinelEnv


ACTION_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_prompt(observation: dict) -> str:
    return (
        "You are the SENTINEL orchestrator. Choose one JSON action.\n"
        f"Task: {observation['task_type']}\n"
        f"Subtask: {observation['current_subtask']}\n"
        f"Stakes: {observation['stakes_level']:.2f}\n"
        f"Trust: {json.dumps(observation['trust_snapshot'], sort_keys=True)}\n"
        "Valid action_type values: delegate, verify, solve_independently, skip.\n"
        "Return JSON with action_type and optional specialist_id."
    )


def parse_action(text: str, observation: dict) -> dict:
    match = ACTION_RE.search(text or "")
    payload = {}
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = {}

    action_type = payload.get("action_type", "delegate")
    specialist_id = payload.get("specialist_id")
    if action_type in ("delegate", "verify") and specialist_id not in observation["available_specialists"]:
        specialist_id = max(
            observation["available_specialists"],
            key=lambda sid: observation["trust_snapshot"].get(sid, 0.5),
        )
    if action_type == "solve_independently":
        specialist_id = None

    return {
        "session_id": observation["session_id"],
        "task_type": observation["task_type"],
        "action_type": action_type,
        "specialist_id": specialist_id,
        "subtask_response": "SELF_SOLVED" if action_type == "solve_independently" else None,
        "reasoning": payload.get("reasoning", "parsed-training-action"),
    }


def dry_run_rollouts(episodes: int, seed: int) -> dict:
    rng = random.Random(seed)
    scores = []
    for idx in range(episodes):
        env = SentinelEnv()
        result = env.reset(task_type="task3", seed=seed + idx)
        while not result["done"]:
            obs = result["observation"]
            specialist = max(obs["available_specialists"], key=lambda sid: obs["trust_snapshot"].get(sid, 0.5))
            action = {
                "session_id": obs["session_id"],
                "task_type": obs["task_type"],
                "action_type": "verify" if obs["stakes_level"] >= 0.70 and rng.random() < 0.5 else "delegate",
                "specialist_id": specialist,
                "subtask_response": None,
                "reasoning": "dry-run heuristic",
            }
            result = env.step(action)
        scores.append(result["info"]["score"])
    return {"episodes": episodes, "avg_score": round(sum(scores) / max(1, len(scores)), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL GRPO training harness.")
    parser.add_argument("--dry-run", action="store_true", help="Run local rollouts without GPU dependencies.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(dry_run_rollouts(args.episodes, args.seed), indent=2))
        return

    try:
        import trl  # noqa: F401
        import unsloth  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are not installed. Run with --dry-run locally, "
            "or install the pyproject training extras on the finale GPU machine."
        ) from exc

    raise SystemExit(
        "GPU training hook is ready. Wire GRPOTrainer here using build_prompt(), "
        "parse_action(), and SentinelEnv.step() as the reward source."
    )


if __name__ == "__main__":
    main()
