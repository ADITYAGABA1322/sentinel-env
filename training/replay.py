from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from environment import SentinelEnv
from mission_context import build_orchestrator_prompt
from sentinel_config import ADVERSARIAL_AWARENESS_STAKES


ACTION_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_replay(path: str | Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Load trained action replay keyed by (task_type, seed, step_count)."""
    table: dict[tuple[str, int, int], dict[str, Any]] = {}
    replay_path = Path(path)
    if not replay_path.exists():
        return table

    for line in replay_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["task_type"]), int(row["seed"]), int(row["step"]))
        table[key] = dict(row["action"])
    return table


class TrainedReplayPolicy:
    """
    Policy callable for training/evaluate.py.

    The Space does not need a GPU at runtime. It looks up a recorded action for
    the current task, seed, and step. Missing rows fall back to the heuristic so
    demos remain robust for unseen seeds.
    """

    def __init__(self, replay_path: str | Path) -> None:
        self.replay_path = Path(replay_path)
        self._table = load_replay(self.replay_path)
        self._task_type = "task3"
        self._seed = 0

    def set_episode(self, task_type: str, seed: int) -> None:
        self._task_type = task_type
        self._seed = seed

    def __call__(self, env: SentinelEnv, obs: dict, rng: random.Random) -> dict:
        key = (self._task_type, self._seed, int(obs.get("step_count", 0)))
        action = dict(self._table.get(key) or {})
        if not action:
            action = heuristic_action(obs)
            action["reasoning"] = "trained replay miss; heuristic fallback"
            action["replay_miss"] = True

        action["session_id"] = obs["session_id"]
        action["task_type"] = obs["task_type"]
        return sanitize_action(action, obs)


def replay_trained_policy(replay_path: str | Path) -> TrainedReplayPolicy:
    return TrainedReplayPolicy(replay_path)


def record_trained_actions(
    adapter_path: str | Path,
    base_model: str,
    tasks: Iterable[str],
    seeds: Iterable[int],
    out_path: str | Path = "outputs/trained_policy_replay.jsonl",
    max_new_tokens: int = 192,
) -> Path:
    """
    Roll out a trained LoRA policy and write replay JSONL.

    In Colab, this loads the trained adapter and samples model actions. Locally,
    if training dependencies or adapter files are unavailable, it falls back to
    the heuristic policy and marks rows with model_source="heuristic_fallback".
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    generator = _load_generator(adapter_path, base_model, max_new_tokens)
    rows: list[dict[str, Any]] = []
    for task_type in tasks:
        for seed in seeds:
            env = SentinelEnv()
            result = env.reset(task_type=task_type, seed=int(seed))
            while not result["done"]:
                obs = result["observation"]
                if generator is None:
                    action = heuristic_action(obs)
                    model_source = "heuristic_fallback"
                else:
                    text = generator(build_orchestrator_prompt(obs))
                    action = parse_action(text, obs)
                    model_source = "trained_lora"
                action["reasoning"] = action.get("reasoning") or model_source
                rows.append(
                    {
                        "task_type": task_type,
                        "seed": int(seed),
                        "scenario_id": obs.get("scenario_id"),
                        "step": int(obs.get("step_count", 0)),
                        "action": {
                            key: value
                            for key, value in action.items()
                            if key in {"action_type", "specialist_id", "subtask_response", "reasoning"}
                        },
                        "model_source": model_source,
                    }
                )
                result = env.step(action)

    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def _load_generator(adapter_path: str | Path, base_model: str, max_new_tokens: int):
    adapter = Path(adapter_path)
    if not adapter.exists():
        return None
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception:
        return None

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    return generate


def parse_action(text: str, obs: dict) -> dict[str, Any]:
    match = ACTION_RE.search(text or "")
    payload: dict[str, Any] = {}
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = {}
    return sanitize_action(payload, obs)


def sanitize_action(payload: dict[str, Any], obs: dict) -> dict[str, Any]:
    action_type = payload.get("action_type", "delegate")
    if action_type not in {"delegate", "verify", "solve_independently", "skip"}:
        action_type = "delegate"

    specialist_id = payload.get("specialist_id")
    if action_type in {"delegate", "verify"} and specialist_id not in obs["available_specialists"]:
        specialist_id = max(
            obs["available_specialists"],
            key=lambda sid: obs["trust_snapshot"].get(sid, 0.5),
        )
    if action_type in {"solve_independently", "skip"}:
        specialist_id = None

    return {
        "session_id": obs["session_id"],
        "task_type": obs["task_type"],
        "action_type": action_type,
        "specialist_id": specialist_id,
        "subtask_response": "SELF_SOLVED" if action_type == "solve_independently" else None,
        "reasoning": payload.get("reasoning", "trained replay action"),
    }


def heuristic_action(obs: dict) -> dict[str, Any]:
    trust = obs["trust_snapshot"]
    specialist = max(obs["available_specialists"], key=lambda sid: trust.get(sid, 0.5))
    action_type = (
        "verify"
        if obs["stakes_level"] >= ADVERSARIAL_AWARENESS_STAKES and trust.get(specialist, 0.5) < 0.70
        else "delegate"
    )
    return {
        "session_id": obs["session_id"],
        "task_type": obs["task_type"],
        "action_type": action_type,
        "specialist_id": specialist,
        "subtask_response": None,
        "reasoning": "heuristic replay baseline",
    }
