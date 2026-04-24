from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import zlib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import SentinelEnv, _GROUND_TRUTH_RELIABILITY
from sentinel_config import ADVERSARIAL_AWARENESS_STAKES


Policy = Callable[[SentinelEnv, dict, random.Random], dict]

POLICIES: dict[str, Policy] = {}


def random_policy(env: SentinelEnv, obs: dict, rng: random.Random) -> dict:
    specialist = rng.choice(obs["available_specialists"])
    roll = rng.random()
    if roll < 0.65:
        action_type = "delegate"
    elif roll < 0.85:
        action_type = "verify"
    elif roll < 0.95:
        action_type = "solve_independently"
        specialist = None
    else:
        action_type = "skip"
        specialist = None
    return _action(obs, action_type, specialist)


def heuristic_policy(env: SentinelEnv, obs: dict, rng: random.Random) -> dict:
    trust = obs["trust_snapshot"]
    specialist = max(obs["available_specialists"], key=lambda sid: trust.get(sid, 0.5))
    action_type = (
        "verify"
        if obs["stakes_level"] >= ADVERSARIAL_AWARENESS_STAKES and trust.get(specialist, 0.5) < 0.65
        else "delegate"
    )
    return _action(obs, action_type, specialist)


def oracle_lite_policy(env: SentinelEnv, obs: dict, rng: random.Random) -> dict:
    reliability = env._pool.public_ground_truth_reliability(_GROUND_TRUTH_RELIABILITY)
    if obs["task_type"] == "task3" and obs["stakes_level"] >= ADVERSARIAL_AWARENESS_STAKES:
        return _action(obs, "verify", env._pool.adversarial_slot)
    specialist = max(obs["available_specialists"], key=lambda sid: reliability.get(sid, 0.5))
    return _action(obs, "delegate", specialist)


def _action(obs: dict, action_type: str, specialist_id: str | None) -> dict:
    return {
        "session_id": obs["session_id"],
        "task_type": obs["task_type"],
        "action_type": action_type,
        "specialist_id": specialist_id,
        "subtask_response": "SELF_SOLVED" if action_type == "solve_independently" else None,
        "reasoning": f"{action_type} via {specialist_id or 'SELF'}",
    }


def run_episode(policy_name: str, policy: Policy, task_type: str, seed: int) -> dict:
    rng = random.Random(seed)
    env = SentinelEnv()
    result = env.reset(task_type=task_type, seed=seed)
    rewards: list[float] = []

    while not result["done"]:
        action = policy(env, result["observation"], rng)
        result = env.step(action)
        rewards.append(result["reward"]["value"])

    info = result["info"]
    breakdown = result["reward"]["signal_breakdown"]
    detections = info.get("adversarial_detections", 0)
    poisonings = info.get("adversarial_poisonings", 0)
    total_adversarial = detections + poisonings
    detection_rate = detections / total_adversarial if total_adversarial else breakdown.get("detection_rate", 1.0)

    return {
        "policy": policy_name,
        "task_type": task_type,
        "seed": seed,
        "steps": info.get("step_count", 0),
        "score": round(info.get("score", 0.0), 4),
        "total_reward": round(info.get("total_reward", 0.0), 4),
        "completion_rate": round(info.get("completion_rate", 0.0), 4),
        "detection_rate": round(detection_rate, 4),
        "trust_calibration": round(breakdown.get("trust_calibration", 0.0), 4),
        "adversarial_detections": detections,
        "adversarial_poisonings": poisonings,
        "status": "failed" if info.get("forced_end") else "completed",
        "rewards": [round(value, 4) for value in rewards],
    }


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["policy"], []).append(row)

    summary = {}
    for policy_name, policy_rows in grouped.items():
        n = len(policy_rows)
        summary[policy_name] = {
            "episodes": n,
            "avg_score": _avg(policy_rows, "score"),
            "avg_completion_rate": _avg(policy_rows, "completion_rate"),
            "avg_detection_rate": _avg(policy_rows, "detection_rate"),
            "avg_trust_calibration": _avg(policy_rows, "trust_calibration"),
            "avg_steps": _avg(policy_rows, "steps"),
        }
    return summary


def _avg(rows: list[dict], key: str) -> float:
    return round(sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows)), 4)


def summarize_by_task(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["task_type"], []).append(row)
    return {task: summarize(task_rows) for task, task_rows in sorted(grouped.items())}


FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def write_baseline_chart(payload: dict, path: Path) -> None:
    """Write a dependency-free PNG chart for README and onsite demos."""
    by_task = payload["by_task"]
    tasks = list(by_task.keys())
    policies = [name for name in ("random", "heuristic", "oracle_lite") if any(name in by_task[t] for t in tasks)]
    colors = {
        "random": (239, 68, 68),
        "heuristic": (59, 130, 246),
        "oracle_lite": (16, 185, 129),
    }
    labels = {"random": "RANDOM", "heuristic": "HEURISTIC", "oracle_lite": "ORACLE LITE"}

    width, height = 1200, 720
    canvas = bytearray([255, 255, 255] * width * height)

    def rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        for y in range(y0, y1):
            row = y * width * 3
            for x in range(x0, x1):
                idx = row + x * 3
                canvas[idx : idx + 3] = bytes(color)

    def text(x: int, y: int, value: str, color: tuple[int, int, int] = (20, 20, 20), scale: int = 2) -> None:
        cursor = x
        for ch in value.upper():
            glyph = FONT_5X7.get(ch, FONT_5X7[" "])
            for gy, line in enumerate(glyph):
                for gx, bit in enumerate(line):
                    if bit == "1":
                        rect(cursor + gx * scale, y + gy * scale, cursor + (gx + 1) * scale, y + (gy + 1) * scale, color)
            cursor += 6 * scale

    def line_h(y: int, x0: int, x1: int, color: tuple[int, int, int]) -> None:
        rect(x0, y, x1, y + 1, color)

    def line_v(x: int, y0: int, y1: int, color: tuple[int, int, int]) -> None:
        rect(x, y0, x + 1, y1, color)

    margin_left, margin_top, margin_right, margin_bottom = 100, 115, 40, 115
    plot_x0, plot_y0 = margin_left, margin_top
    plot_x1, plot_y1 = width - margin_right, height - margin_bottom
    plot_w, plot_h = plot_x1 - plot_x0, plot_y1 - plot_y0

    text(50, 28, "SENTINEL BASELINE COMPARISON", (17, 24, 39), 3)
    text(52, 70, "EPISODE SCORE 0.0 TO 1.0 - RANDOM VS TRUST WEIGHTED VS ORACLE LITE", (75, 85, 99), 2)

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(plot_y1 - tick * plot_h)
        line_h(y, plot_x0, plot_x1, (226, 232, 240))
        text(32, y - 7, f"{tick:.2f}", (100, 116, 139), 2)
    line_v(plot_x0, plot_y0, plot_y1, (148, 163, 184))
    line_h(plot_y1, plot_x0, plot_x1, (148, 163, 184))

    group_w = plot_w / max(1, len(tasks))
    bar_w = max(34, min(76, int((group_w - 80) / max(1, len(policies)))))
    for task_idx, task in enumerate(tasks):
        group_center = int(plot_x0 + group_w * task_idx + group_w / 2)
        start_x = group_center - int((len(policies) * bar_w + (len(policies) - 1) * 18) / 2)
        for policy_idx, policy in enumerate(policies):
            value = float(by_task[task].get(policy, {}).get("avg_score", 0.0))
            x0 = start_x + policy_idx * (bar_w + 18)
            y0 = int(plot_y1 - value * plot_h)
            rect(x0 + 3, y0 + 3, x0 + bar_w + 3, plot_y1 + 3, (203, 213, 225))
            rect(x0, y0, x0 + bar_w, plot_y1, colors[policy])
            text(x0 - 4, max(plot_y0 - 2, y0 - 24), f"{value:.2f}", (15, 23, 42), 2)
        text(group_center - 36, plot_y1 + 30, task.upper(), (15, 23, 42), 2)

    legend_x, legend_y = 780, 32
    for idx, policy in enumerate(policies):
        x = legend_x
        y = legend_y + idx * 24
        rect(x, y, x + 16, y + 16, colors[policy])
        text(x + 24, y + 1, labels[policy], (51, 65, 85), 2)

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path, width, height, canvas)


def _write_png(path: Path, width: int, height: int, rgb: bytearray) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = []
    stride = width * 3
    for y in range(height):
        rows.append(b"\x00" + bytes(rgb[y * stride : (y + 1) * stride]))
    raw = b"".join(rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SENTINEL policies.")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per policy.")
    parser.add_argument("--task", default="task3", choices=["task1", "task2", "task3", "all"])
    parser.add_argument("--out", default="outputs/evaluation_results.json")
    parser.add_argument("--plot", default="outputs/baseline_comparison.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    policies: dict[str, Policy] = {
        "random": random_policy,
        "heuristic": heuristic_policy,
        "oracle_lite": oracle_lite_policy,
    }

    tasks = ["task1", "task2", "task3"] if args.task == "all" else [args.task]
    rows = []
    for task_type in tasks:
        for policy_name, policy in policies.items():
            for seed in range(args.episodes):
                rows.append(run_episode(policy_name, policy, task_type, seed))

    payload = {
        "task": args.task,
        "tasks": tasks,
        "episodes_per_policy": args.episodes,
        "summary": summarize(rows),
        "by_task": summarize_by_task(rows),
        "episodes": rows,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    if not args.no_plot:
        chart_path = ROOT / args.plot
        write_baseline_chart(payload, chart_path)
        payload["chart"] = str(chart_path.relative_to(ROOT))
        out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps({"summary": payload["summary"], "by_task": payload["by_task"], "chart": payload.get("chart")}, indent=2))


if __name__ == "__main__":
    main()
