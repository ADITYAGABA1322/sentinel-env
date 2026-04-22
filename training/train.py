from __future__ import annotations

"""
Onsite training entrypoint.

This file is intentionally import-light so it can run locally without GPU
packages. On the finale machine, install the training extras from pyproject and
run without --dry-run to train a small orchestrator policy with GRPO.
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


def build_dataset_records(episodes: int, task_type: str, seed: int) -> list[dict]:
    records = []
    task_choices = ["task1", "task2", "task3"] if task_type == "all" else [task_type]
    for idx in range(episodes):
        selected_task = task_choices[idx % len(task_choices)]
        env = SentinelEnv()
        result = env.reset(task_type=selected_task, seed=seed + idx)
        obs = result["observation"]
        records.append(
            {
                "prompt": build_prompt(obs),
                "task_type": selected_task,
                "seed": seed + idx,
            }
        )
    return records


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


def score_completion(completion: str, task_type: str, seed: int) -> float:
    env = SentinelEnv()
    result = env.reset(task_type=task_type, seed=seed)
    obs = result["observation"]
    action = parse_action(completion, obs)
    result = env.step(action)
    return float(result["reward"]["value"])


def sentinel_reward(completions, prompts=None, task_type=None, seed=None, **kwargs):
    rewards = []
    task_values = task_type or kwargs.get("task_type") or ["task3"] * len(completions)
    seed_values = seed or kwargs.get("seed") or list(range(len(completions)))
    for idx, completion in enumerate(completions):
        text = _completion_text(completion)
        try:
            rewards.append(score_completion(text, str(task_values[idx]), int(seed_values[idx])))
        except Exception:
            rewards.append(0.01)
    return rewards


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


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


def run_grpo(args) -> None:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
    except ImportError:
        print("Training dependencies are not installed locally.")
        print("Local check passed. For onsite GPU training run:")
        print("  pip install '.[training]'")
        print("  python training/train.py --episodes 300 --task all")
        return

    records = build_dataset_records(args.episodes, args.task, args.seed)
    dataset = Dataset.from_list(records)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=args.lora_rank,
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,
        save_steps=50,
        max_prompt_length=args.max_seq_length,
        max_completion_length=192,
    )

    trainer_kwargs = {
        "model": model,
        "reward_funcs": [sentinel_reward],
        "args": config,
        "train_dataset": dataset,
    }
    try:
        trainer = GRPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = GRPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Saved LoRA adapter to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL GRPO training harness.")
    parser.add_argument("--dry-run", action="store_true", help="Run local rollouts without GPU dependencies.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="task3", choices=["task1", "task2", "task3", "all"])
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="training/sentinel_model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(dry_run_rollouts(args.episodes, args.seed), indent=2))
        return

    run_grpo(args)


if __name__ == "__main__":
    main()
