"""
training/plot_training.py
=========================

Plots SENTINEL-Σ training curves from:
- a local JSONL metrics log produced by `training/train_sigma.py`, OR
- a wandb run (best-effort if wandb is installed + run id is provided).

Outputs (PNG) saved to outputs/:
- training_reward_curve.png
- detection_rate_curve.png
- atlas_growth.png
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CURRICULUM = [
    (0, 50, "task1"),
    (50, 150, "task2"),
    (150, 300, "task3"),
]


def _load_jsonl(path: Path) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_from_wandb(run_path: str) -> List[dict[str, Any]]:
    """
    Load history rows from a wandb run path like 'entity/project/run_id'.
    """
    import wandb  # type: ignore

    api = wandb.Api()
    run = api.run(run_path)
    rows: List[dict[str, Any]] = []
    for r in run.scan_history():
        rows.append(dict(r))
    return rows


def _phase_for_step(step: int) -> str:
    for lo, hi, task in CURRICULUM:
        if lo <= step < hi:
            return task
    return "task3"


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(os.environ.get("METRICS_PATH", "outputs/sigma_metrics.jsonl"))
    wandb_run = os.environ.get("WANDB_RUN", "").strip()

    rows: List[dict[str, Any]]
    if wandb_run:
        try:
            rows = _load_from_wandb(wandb_run)
        except Exception as e:
            raise SystemExit(f"Failed to load wandb run {wandb_run!r}: {e}")
    else:
        if not metrics_path.exists():
            raise SystemExit(f"Metrics file not found: {metrics_path}")
        rows = _load_jsonl(metrics_path)

    # Normalize and sort by step
    cleaned = []
    for r in rows:
        if "step" not in r:
            continue
        try:
            step = int(r["step"])
        except Exception:
            continue
        cleaned.append(r | {"step": step})
    cleaned.sort(key=lambda x: x["step"])

    if not cleaned:
        raise SystemExit("No usable metrics rows found.")

    steps = [r["step"] for r in cleaned]
    reward_mean = [float(r.get("reward/mean", 0.0) or 0.0) for r in cleaned]
    det_rate = [float(r.get("detection_rate", 0.0) or 0.0) for r in cleaned]
    archetypes = [int(r.get("atlas_archetypes_count", 0) or 0) for r in cleaned]

    import matplotlib.pyplot as plt

    def shade_curriculum(ax):
        colors = {"task1": "#0ea5e9", "task2": "#a78bfa", "task3": "#22c55e"}
        for lo, hi, task in CURRICULUM:
            ax.axvspan(lo, hi, alpha=0.08, color=colors.get(task, "#94a3b8"), label=f"{task} phase")

    # 1) Reward curve
    fig, ax = plt.subplots(figsize=(10, 4.5))
    shade_curriculum(ax)
    ax.plot(steps, reward_mean, color="#e5eef8", linewidth=2, label="reward/mean")
    ax.set_title("SENTINEL-Σ training reward over steps")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean reward")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "training_reward_curve.png", dpi=160)
    plt.close(fig)

    # 2) Detection rate curve
    fig, ax = plt.subplots(figsize=(10, 4.5))
    shade_curriculum(ax)
    ax.plot(steps, det_rate, color="#f59e0b", linewidth=2, label="detection_rate")
    ax.axhline(0.05, color="#94a3b8", linestyle="--", linewidth=1.5, label="random baseline (0.05)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Adversary detection rate over training steps")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Detection rate")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "detection_rate_curve.png", dpi=160)
    plt.close(fig)

    # 3) Atlas growth
    fig, ax = plt.subplots(figsize=(10, 4.5))
    shade_curriculum(ax)
    ax.plot(steps, archetypes, color="#22c55e", linewidth=2, label="atlas_archetypes_count")
    ax.set_title("Behavioral atlas growth (archetypes discovered)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Archetypes count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "atlas_growth.png", dpi=160)
    plt.close(fig)

    print("Saved plots to outputs/:")
    print("- training_reward_curve.png")
    print("- detection_rate_curve.png")
    print("- atlas_growth.png")


if __name__ == "__main__":
    main()

