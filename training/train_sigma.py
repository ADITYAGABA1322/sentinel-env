"""
training/train_sigma.py
======================

SENTINEL-Σ training script using **HuggingFace TRL GRPO** + **Unsloth**.

Model
-----
- unsloth/Qwen2.5-1.5B-Instruct
- 4-bit quantization

Curriculum (training steps)
---------------------------
- steps 0–50     -> task1
- steps 50–150   -> task2
- steps 150–300  -> task3

Environment
-----------
Talks to the SENTINEL FastAPI env via HTTP:
- ENV_URL from environment (default: http://localhost:7860)
- POST /reset
- POST /step?session_id=

Reward function contract (required)
-----------------------------------
reward_fn must:
1) Parse model completion as JSON action: {action_type, specialist_id, reasoning}
2) POST to /step?session_id=
3) Extract reward.value from response
4) Handle parse errors/timeouts/HTTP errors -> 0.01 penalty

Logging
-------
Tracks these metrics per training step (wandb if available, else prints; always JSONL):
- reward/mean, reward/std
- reward/task1_mean, reward/task2_mean, reward/task3_mean
- detection_rate (fraction of episodes where adversary was identified before step 10)
- atlas_archetypes_count (best-effort; remote env may not expose)
- curriculum_phase
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx


CURRICULUM = [
    (0, 50, "task1"),
    (50, 150, "task2"),
    (150, 300, "task3"),
]


def curriculum_phase(step: int) -> str:
    for lo, hi, task in CURRICULUM:
        if lo <= step < hi:
            return task
    return "task3"


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)


def _extract_observation_from_prompt(prompt: str) -> dict[str, Any]:
    """
    Extract the observation JSON blob from our prompt template.
    """
    marker = "OBSERVATION_JSON:\n"
    idx = prompt.rfind(marker)
    if idx < 0:
        return {}
    raw = prompt[idx + len(marker) :].strip()
    # raw should be valid JSON
    return json.loads(raw)


def _parse_action_json(completion: str) -> dict[str, Any]:
    """
    Parse a completion that should be a single JSON object.
    Strips common markdown fences if present.
    """
    text = completion.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("completion is not a JSON object")
    return obj


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (_mean([(x - m) ** 2 for x in xs])) ** 0.5


@dataclass
class EpisodeHandle:
    session_id: str
    task_type: str
    decision_step_count: int = 0
    detected_before_10: bool = False


class SentinelHTTP:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def reset(self, task_type: str, seed: Optional[int] = None) -> dict[str, Any]:
        r = self.client.post("/reset", json={"task_type": task_type, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = self.client.post(f"/step?session_id={session_id}", json=payload)
        r.raise_for_status()
        return r.json()


class PromptStream(Iterable[dict[str, Any]]):
    """
    Infinite stream of prompts for GRPO.

    Each sample is a single decision-step interaction:
    - We call /reset to start a fresh episode and obtain an observation.
    - The prompt includes that observation as clean JSON.
    - We rely on the reward_fn to parse session_id from the prompt and call /step.
    """

    def __init__(self, env: SentinelHTTP, start_step: int = 0) -> None:
        self.env = env
        self.global_step = start_step

    def __iter__(self):
        while True:
            task = curriculum_phase(self.global_step)
            reset = self.env.reset(task_type=task, seed=self.global_step)
            obs = reset.get("observation", {})

            # Ensure prompt mentions the required fields explicitly.
            prompt = (
                "You are the SENTINEL orchestrator.\n"
                "You must choose the next action to take.\n\n"
                "Respond with a JSON object only: {action_type, specialist_id, reasoning}\n"
                "Valid action_type: delegate | verify | solve_independently | skip\n"
                "Valid specialist_id: S0 | S1 | S2 | S3 | S4 (required for delegate/verify)\n\n"
                "The observation includes trust, behavioral_fingerprints, and may include atlas_priors.\n"
                "Use them to detect adversarial patterns and route safely.\n\n"
                "OBSERVATION_JSON:\n"
                f"{_safe_json_dumps(obs)}"
            )

            yield {
                "prompt": prompt,
                "curriculum_phase": task,
            }
            self.global_step += 1


class RewardAndMetrics:
    """
    Callable reward function for TRL GRPOTrainer.
    Also maintains metric state and logs to JSONL / wandb.
    """

    def __init__(self, env: SentinelHTTP, log_path: Path, use_wandb: bool) -> None:
        self.env = env
        self.log_path = log_path
        self.use_wandb = use_wandb

        self._session_cache: Dict[str, EpisodeHandle] = {}
        self._last_log_t = 0.0

        self._wandb = None
        if use_wandb:
            try:
                import wandb  # type: ignore

                self._wandb = wandb
                if wandb.run is None:
                    wandb.init(project="sentinel-sigma", name=f"sigma-{int(time.time())}")
            except Exception:
                self._wandb = None

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, row: dict[str, Any]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self._wandb is not None:
            try:
                self._wandb.log(row)
            except Exception:
                pass
        else:
            # Keep console noise low: print occasionally or on curriculum boundaries.
            now = time.time()
            if now - self._last_log_t > 2.0 or (row.get("step", 0) in (0, 50, 150, 299)):
                self._last_log_t = now
                print(json.dumps(row, ensure_ascii=False))

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        per_task_rewards: Dict[str, List[float]] = {"task1": [], "task2": [], "task3": []}

        detections = 0
        episodes_seen = 0
        archetypes_count = 0

        for prompt, completion in zip(prompts, completions):
            try:
                obs = _extract_observation_from_prompt(prompt)
                session_id = str(obs.get("session_id", "")).strip()
                task_type = str(obs.get("task_type", "task3"))
                step_count = int(obs.get("step_count", 0) or 0)
                phase = task_type

                if not session_id:
                    raise ValueError("prompt missing session_id in observation")

                # Track episode handle (best-effort; remote env cleans up when done).
                h = self._session_cache.get(session_id)
                if h is None:
                    h = EpisodeHandle(session_id=session_id, task_type=task_type, decision_step_count=0)
                    self._session_cache[session_id] = h
                    episodes_seen += 1

                action_obj = _parse_action_json(completion)
                action_type = str(action_obj.get("action_type", "")).strip()
                specialist_id = action_obj.get("specialist_id")
                reasoning = action_obj.get("reasoning")

                # Build /step payload (server expects full action contract).
                payload = {
                    "session_id": session_id,
                    "task_type": task_type,
                    "action_type": action_type,
                    "specialist_id": specialist_id,
                    "subtask_response": None,
                    "reasoning": reasoning,
                }

                # Minimal schema guardrail
                if action_type not in ("delegate", "verify", "solve_independently", "skip"):
                    raise ValueError(f"invalid action_type={action_type!r}")
                if action_type in ("delegate", "verify") and (specialist_id not in ("S0", "S1", "S2", "S3", "S4")):
                    raise ValueError("delegate/verify requires specialist_id in S0..S4")

                resp = self.env.step(session_id=session_id, payload=payload)
                r = float(resp["reward"]["value"])
                reason = str(resp["reward"].get("reason", ""))

                # Detection metric: count episodes where adversary was identified before step 10.
                # We approximate identification using the env's reward reason string on task3:
                # "Adversarial detected ..." is emitted on high-stakes verify/self-solve.
                h.decision_step_count += 1
                if (not h.detected_before_10) and (step_count < 10) and ("Adversarial detected" in reason):
                    h.detected_before_10 = True

                if h.detected_before_10:
                    detections += 1

                rewards.append(r)
                per_task_rewards.get(phase, per_task_rewards["task3"]).append(r)

                # atlas archetypes count: remote env does not expose; keep 0 unless present.
                # If your env later adds obs["atlas_summary"]["archetypes"], this will start working.
                try:
                    a = obs.get("atlas_summary") or {}
                    archetypes_count = max(archetypes_count, len(a.get("archetypes", [])))
                except Exception:
                    pass

            except (json.JSONDecodeError, ValueError, KeyError, httpx.TimeoutException, httpx.HTTPError):
                rewards.append(0.01)

        # Aggregate and log
        step = int(kwargs.get("global_step", kwargs.get("step", -1)) or -1)
        phase = curriculum_phase(step) if step >= 0 else "unknown"
        det_rate = (detections / max(1, episodes_seen)) if episodes_seen else 0.0

        row = {
            "step": step,
            "curriculum_phase": phase,
            "reward/mean": round(_mean(rewards), 4),
            "reward/std": round(_std(rewards), 4),
            "reward/task1_mean": round(_mean(per_task_rewards["task1"]), 4),
            "reward/task2_mean": round(_mean(per_task_rewards["task2"]), 4),
            "reward/task3_mean": round(_mean(per_task_rewards["task3"]), 4),
            "detection_rate": round(det_rate, 4),
            "atlas_archetypes_count": int(archetypes_count),
        }
        if step >= 0:
            self._log(row)

        return rewards


def main() -> None:
    env_url = os.environ.get("ENV_URL", "http://localhost:7860").strip()
    total_steps = int(os.environ.get("TOTAL_STEPS", "300"))
    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = SentinelHTTP(env_url, timeout_s=30.0)
        env.client.get("/health").raise_for_status()
    except Exception as e:
        raise SystemExit(f"Could not connect to ENV_URL={env_url!r}: {e}")

    # Optional wandb (if installed + configured).
    use_wandb = os.environ.get("USE_WANDB", "1").strip() != "0"
    log_path = Path("outputs") / "sigma_metrics.jsonl"

    # Imports (deferred) so script provides actionable errors.
    try:
        from datasets import IterableDataset  # type: ignore
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
        from transformers import TrainingArguments  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing training deps. Install at least: trl, transformers, datasets, accelerate, wandb (optional).\n"
            f"Import error: {e}"
        )

    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Unsloth not available. Install unsloth and its CUDA-compatible deps.\n"
            f"Import error: {e}"
        )

    # Build model + tokenizer (4-bit quant)
    model_name = "unsloth/Qwen2.5-1.5B-Instruct"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Dataset: infinite prompt stream wrapped as HF IterableDataset
    stream = PromptStream(env=env, start_step=0)
    train_dataset = IterableDataset.from_generator(lambda: iter(stream))

    # Reward function
    rewarder = RewardAndMetrics(env=env, log_path=log_path, use_wandb=use_wandb)

    # GRPO config / args
    # Note: TRL APIs change; we keep the configuration minimal and rely on defaults.
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", "2")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACC", "4")),
        learning_rate=float(os.environ.get("LR", "2e-5")),
        max_steps=total_steps,
        logging_steps=1,
        save_steps=50,
        save_total_limit=999,
        bf16=False,
        fp16=True,
        report_to=["wandb"] if (rewarder._wandb is not None) else [],
    )

    grpo_cfg = GRPOConfig(
        max_prompt_length=1536,
        max_completion_length=256,
        num_generations=int(os.environ.get("NUM_GENERATIONS", "2")),
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_funcs=[rewarder],
        grpo_config=grpo_cfg,
    )

    trainer.train()

    # Save merged model
    final_dir = out_dir / "final_merged"
    final_dir.mkdir(parents=True, exist_ok=True)
    try:
        FastLanguageModel.for_inference(model)
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
    except Exception:
        # Fallback to trainer save
        trainer.save_model(str(final_dir))

    print(f"Saved final merged model to: {final_dir}")


if __name__ == "__main__":
    main()

