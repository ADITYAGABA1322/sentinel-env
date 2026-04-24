from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import SentinelEnv, _GROUND_TRUTH_RELIABILITY
from mission_context import build_orchestrator_prompt, mission_for_task, problem_statement


Policy = Callable[[SentinelEnv, dict, random.Random], dict]


@dataclass
class TraceRow:
    step_count: int
    subtask_index: int
    stakes: float
    action: str
    reward: float
    score: float
    completed: int
    remaining: int
    trust: dict[str, float]
    reason: str
    detections: int
    poisonings: int


def blind_trust_policy(env: SentinelEnv, obs: dict, rng: random.Random) -> dict:
    """Before SENTINEL: always trust the currently highest-trust specialist."""
    trust = obs["trust_snapshot"]
    specialist = max(obs["available_specialists"], key=lambda sid: trust.get(sid, 0.5))
    return action(obs, "delegate", specialist, f"blind-trust best={specialist}")


def sentinel_heuristic_policy(env: SentinelEnv, obs: dict, rng: random.Random) -> dict:
    """
    Simple SENTINEL-style baseline.
    It still has flaws, but it shows the intended skill: verify when risk rises.
    """
    trust = obs["trust_snapshot"]
    specialist = max(obs["available_specialists"], key=lambda sid: trust.get(sid, 0.5))
    score = trust.get(specialist, 0.5)
    action_type = "verify" if obs["stakes_level"] >= 0.70 and score < 0.65 else "delegate"
    return action(
        obs,
        action_type,
        specialist,
        f"sentinel-heuristic best={specialist} trust={score:.3f} stakes={obs['stakes_level']:.2f}",
    )


def oracle_lite_policy(env: SentinelEnv, obs: dict, rng: random.Random) -> dict:
    """
    Upper-bound policy used for demos and evaluator comparison.
    It uses hidden builder-only info, so it is NOT a deployable policy.
    """
    reliability = env._pool.public_ground_truth_reliability(_GROUND_TRUTH_RELIABILITY)
    if obs["task_type"] == "task3" and obs["stakes_level"] >= 0.70:
        return action(obs, "verify", env._pool.adversarial_slot, "oracle-lite verifies adversarial slot")
    specialist = max(obs["available_specialists"], key=lambda sid: reliability.get(sid, 0.5))
    return action(obs, "delegate", specialist, f"oracle-lite best={specialist}")


POLICIES: dict[str, Policy] = {
    "blind": blind_trust_policy,
    "heuristic": sentinel_heuristic_policy,
    "oracle": oracle_lite_policy,
}


def action(obs: dict, action_type: str, specialist_id: str | None, reason: str) -> dict:
    return {
        "session_id": obs["session_id"],
        "task_type": obs["task_type"],
        "action_type": action_type,
        "specialist_id": specialist_id,
        "subtask_response": "SELF_SOLVED" if action_type == "solve_independently" else None,
        "reasoning": reason,
    }


def compact_reset(result: dict) -> dict:
    obs = result["observation"]
    return {
        "session_id": obs["session_id"],
        "scenario_id": obs["scenario_id"],
        "task_type": obs["task_type"],
        "current_subtask": obs["current_subtask"],
        "available_specialists": obs["available_specialists"],
        "trust_snapshot": obs["trust_snapshot"],
        "stakes_level": obs["stakes_level"],
        "step_count": obs["step_count"],
        "max_steps": obs["max_steps"],
        "done": result["done"],
        "reward": result["reward"],
    }


def run_episode(
    policy_name: str,
    task_type: str,
    seed: int,
    show_hidden: bool,
    max_rows: int | None,
) -> tuple[SentinelEnv, dict, list[TraceRow]]:
    policy = POLICIES[policy_name]
    rng = random.Random(seed)
    env = SentinelEnv()
    result = env.reset(task_type=task_type, seed=seed)
    rows: list[TraceRow] = []

    print_header(policy_name, task_type, seed)
    print("RESET JSON - compact agent-facing shape")
    print(json.dumps(compact_reset(result), indent=2))
    print()
    print("LLM ORCHESTRATOR PROMPT - first 28 lines")
    prompt_lines = build_orchestrator_prompt(result["observation"]).splitlines()
    print("\n".join(prompt_lines[:28]))
    if len(prompt_lines) > 28:
        print("...")
    print()
    if show_hidden:
        print("BUILDER-ONLY HIDDEN PROFILE - agent never sees this")
        print(json.dumps({
            "public_slot_to_internal_behavior": env._pool.internal_profile(),
            "adversarial_public_slot": env._pool.adversarial_slot,
        }, indent=2))
        print()

    print_trace_header()
    guard = 0
    while not result["done"] and guard < 100:
        obs = result["observation"]
        chosen = policy(env, obs, rng)
        result = env.step(chosen)
        graph_summary = env._graph.summary()
        row = TraceRow(
            step_count=result["info"]["step_count"],
            subtask_index=result["observation"]["subtask_index"],
            stakes=obs["stakes_level"],
            action=f"{chosen['action_type']}:{chosen.get('specialist_id') or 'SELF'}",
            reward=result["reward"]["value"],
            score=result["info"]["score"],
            completed=graph_summary["subtasks_completed"],
            remaining=graph_summary["subtasks_remaining"],
            trust=result["observation"]["trust_snapshot"],
            reason=result["reward"]["reason"],
            detections=graph_summary["adversarial_detections"],
            poisonings=graph_summary["adversarial_poisonings"],
        )
        rows.append(row)
        if max_rows is None or len(rows) <= max_rows:
            print_trace_row(row)
        guard += 1

    if max_rows is not None and len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} more rows hidden by --max-rows")

    print()
    print("FINAL INFO")
    print(json.dumps(result["info"], indent=2))
    print("FINAL REWARD")
    print(json.dumps(result["reward"], indent=2))
    print()
    return env, result, rows


def print_header(policy_name: str, task_type: str, seed: int) -> None:
    problem = problem_statement()["problem"]
    mission = mission_for_task(task_type)
    print("=" * 92)
    print("SENTINEL BACKEND WALKTHROUGH")
    print("=" * 92)
    print(f"policy={policy_name} task={task_type} seed={seed}")
    print()
    print("REAL USER PROMPT EXAMPLE")
    print(problem["real_user_prompt_example"])
    print()
    print("REAL-WORLD MAPPING")
    print(problem["not_a_simple_prompt_solver"])
    print(f"Task mission: {mission['judge_friendly_story']}")
    print("The JSON action is the next internal control move, not the final user answer.")
    print("SENTINEL trains the transferable behavior: trust, verify, recover, finish.")
    print()


def print_trace_header() -> None:
    print("STEP TRACE")
    print(
        "step | node | stake | action          | reward | score | done/rem | adv det/poison | trust snapshot"
    )
    print("-" * 132)


def print_trace_row(row: TraceRow) -> None:
    trust = " ".join(f"{sid}:{score:.3f}" for sid, score in row.trust.items())
    print(
        f"{row.step_count:>4} | {row.subtask_index:>4} | {row.stakes:>5.2f} | "
        f"{row.action:<15} | {row.reward:>6.3f} | {row.score:>5.3f} | "
        f"{row.completed:>2}/{row.completed + row.remaining:<2} | "
        f"{row.detections:>2}/{row.poisonings:<2} | {trust}"
    )
    print(f"     reason: {row.reason}")


def compare_policies(task_type: str, seed: int, show_hidden: bool) -> None:
    mission = mission_for_task(task_type)
    print("=" * 92)
    print("BEFORE / AFTER BACKEND COMPARISON")
    print("=" * 92)
    print("before=blind trust, middle=heuristic trust, target=oracle-lite upper bound")
    print(f"mission={mission['name']} - {mission['real_life_example']}")
    print()
    results = []
    for policy_name in ("blind", "heuristic", "oracle"):
        env = SentinelEnv()
        result = env.reset(task_type=task_type, seed=seed)
        rng = random.Random(seed)
        while not result["done"]:
            chosen = POLICIES[policy_name](env, result["observation"], rng)
            result = env.step(chosen)
        info = result["info"]
        results.append({
            "policy": policy_name,
            "score": info.get("score", 0.0),
            "completion": info.get("completion_rate", 0.0),
            "detections": info.get("adversarial_detections", 0),
            "poisonings": info.get("adversarial_poisonings", 0),
            "steps": info.get("step_count", 0),
            "status": "failed" if info.get("forced_end") else "completed",
        })
        if show_hidden and policy_name == "blind":
            print("Hidden profile for this comparison seed:")
            print(json.dumps({
                "public_slot_to_internal_behavior": env._pool.internal_profile(),
                "adversarial_public_slot": env._pool.adversarial_slot,
            }, indent=2))
            print()

    print("policy    | score | completion | detections | poisonings | steps | status")
    print("-" * 78)
    for item in results:
        print(
            f"{item['policy']:<9} | {item['score']:.3f} | "
            f"{item['completion']:.3f}      | {item['detections']:<10} | "
            f"{item['poisonings']:<10} | {item['steps']:<5} | {item['status']}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain SENTINEL backend behavior from terminal.")
    parser.add_argument("--task", default="task3", choices=["task1", "task2", "task3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", default="heuristic", choices=sorted(POLICIES))
    parser.add_argument("--hide-hidden", action="store_true", help="Do not print builder-only hidden profile.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit printed trace rows.")
    parser.add_argument("--compare", action="store_true", help="Compare blind vs heuristic vs oracle-lite.")
    args = parser.parse_args()

    show_hidden = not args.hide_hidden
    if args.compare:
        compare_policies(args.task, args.seed, show_hidden)
    run_episode(args.policy, args.task, args.seed, show_hidden, args.max_rows)


if __name__ == "__main__":
    main()
