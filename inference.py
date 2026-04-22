"""
SENTINEL — Baseline Inference Script
=====================================
Runs a deterministic heuristic agent against all 3 task types.
Emits [START] / [STEP] / [END] structured logs exactly as required.

Heuristic agent logic:
  - Always delegates to the specialist with highest trust score
  - If stakes >= 0.70 and trust of chosen specialist < 0.60 → verify instead
  - Never skips
  - Never solves independently (too expensive)

Usage:
    python inference.py

Environment variables (optional):
    API_BASE_URL  — OpenAI-compatible endpoint (for LLM agent, not heuristic)
    MODEL_NAME    — model identifier
    HF_TOKEN      — API key
    ENV_URL       — remote env URL (default: in-process)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Try remote env first, fall back to in-process
# ---------------------------------------------------------------------------

ENV_URL = os.environ.get("ENV_URL", "").strip()

if ENV_URL:
    import httpx
    USE_REMOTE = True
else:
    from environment import SentinelEnv
    USE_REMOTE = False


# ---------------------------------------------------------------------------
# Env interface (works both remote and in-process)
# ---------------------------------------------------------------------------

class EnvClient:
    def __init__(self):
        if USE_REMOTE:
            self._client = httpx.Client(base_url=ENV_URL, timeout=30)
        else:
            self._env = SentinelEnv()
        self.session_id: str = ""

    def reset(self, task_type: str, scenario_id: str | None = None, seed: int | None = None) -> dict:
        payload = {"task_type": task_type, "scenario_id": scenario_id, "seed": seed}
        if USE_REMOTE:
            r = self._client.post("/reset", json=payload)
            r.raise_for_status()
            result = r.json()
        else:
            result = self._env.reset(task_type=task_type, scenario_id=scenario_id, seed=seed)
        self.session_id = result["info"]["session_id"]
        return result

    def step(self, action: dict) -> dict:
        action["session_id"] = self.session_id
        if USE_REMOTE:
            r = self._client.post(f"/step?session_id={self.session_id}", json=action)
            r.raise_for_status()
            return r.json()
        else:
            return self._env.step(action)

    def state(self) -> dict:
        if USE_REMOTE:
            r = self._client.get(f"/state?session_id={self.session_id}")
            r.raise_for_status()
            return r.json()
        else:
            return self._env.state(session_id=self.session_id)


# ---------------------------------------------------------------------------
# Heuristic agent
# ---------------------------------------------------------------------------

def heuristic_action(obs: dict, session_id: str) -> dict:
    """
    Deterministic trust-weighted delegation.
    Picks specialist with highest trust. Upgrades to verify at high stakes.
    """
    trust        = obs["trust_snapshot"]          # {"S0": 0.5, ...}
    stakes       = obs["stakes_level"]
    task_type    = obs["task_type"]
    available    = obs["available_specialists"]

    # Pick specialist with highest trust
    best_specialist = max(available, key=lambda s: trust.get(s, 0.5))
    best_trust      = trust.get(best_specialist, 0.5)

    # Upgrade to verify if high stakes AND low trust in best specialist
    if stakes >= 0.70 and best_trust < 0.60:
        action_type = "verify"
    else:
        action_type = "delegate"

    return {
        "session_id":       session_id,
        "task_type":        task_type,
        "action_type":      action_type,
        "specialist_id":    best_specialist,
        "subtask_response": None,
        "reasoning":        f"Trust-weighted: {best_specialist}={best_trust:.3f}, stakes={stakes:.2f}",
    }


# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------

def run_episode(
    client: EnvClient,
    task_type: str,
    scenario_id: str,
    seed: int,
) -> dict:
    result     = client.reset(task_type=task_type, scenario_id=scenario_id, seed=seed)
    session_id = client.session_id

    print(f"[START] task={scenario_id} env=sentinel-env model=heuristic-baseline")

    step_num    = 0
    rewards: list[float] = []
    final_score = 0.0

    while True:
        obs    = result["observation"]
        action = heuristic_action(obs, session_id)

        result    = client.step(action)
        reward    = result["reward"]["value"]
        done      = result["done"]
        step_num += 1
        rewards.append(reward)
        final_score = result["info"].get("score", 0.0)

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

    # Final info
    info = result["info"]
    completion  = info.get("completion_rate", 0.0)
    detections  = info.get("adversarial_detections", 0)
    poisonings  = info.get("adversarial_poisonings", 0)
    trust_snap  = info.get("trust_snapshot", {})
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success=true "
        f"steps={step_num} "
        f"score={final_score:.3f} "
        f"rewards={rewards_str}"
    )

    return {
        "scenario_id":            scenario_id,
        "task_type":              task_type,
        "steps":                  step_num,
        "score":                  round(final_score, 4),
        "total_reward":           round(info.get("total_reward", 0.0), 4),
        "completion_rate":        round(completion, 4),
        "adversarial_detections": detections,
        "adversarial_poisonings": poisonings,
        "final_trust":            trust_snap,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    client = EnvClient()
    all_results = []

    # Run 10 episodes per task type (30 total — fast enough for validation)
    for task_type in ["task1", "task2", "task3"]:
        for i in range(10):
            scenario_id = f"SCN-{task_type.upper()}-{i+1:03d}"
            try:
                result = run_episode(client, task_type, scenario_id, seed=i)
                all_results.append(result)
            except Exception as e:
                print(f"[STEP] step=0 action=error reward=0.00 done=true error={e}")
                print(f"[END] success=false steps=0 score=0.000 rewards=0.00")

    if all_results:
        by_task: dict[str, list] = {"task1": [], "task2": [], "task3": []}
        for r in all_results:
            by_task[r["task_type"]].append(r["score"])

        overall_scores = []
        for task_type, scores in by_task.items():
            if scores:
                overall_scores.extend(scores)

        overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

        out_path = Path("outputs/baseline_scores.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "model":          "heuristic-baseline",
                "total_episodes": len(all_results),
                "avg_score":      round(overall_avg, 4),
                "by_task": {
                    t: {"episodes": len(s), "avg_score": round(sum(s)/len(s), 4)}
                    for t, s in by_task.items() if s
                },
                "episodes": all_results,
            }, f, indent=2)


if __name__ == "__main__":
    main()
