#!/usr/bin/env bash
set -euo pipefail

EPISODES="${EPISODES:-30}"
REPLAY="${REPLAY:-outputs/trained_policy_replay.jsonl}"

python training/evaluate.py \
  --episodes "${EPISODES}" \
  --task all \
  --policies random,heuristic,oracle_lite,trained \
  --replay "${REPLAY}" \
  --out outputs/eval_post.json \
  --plot outputs/charts/baseline_grouped_bars.png

python -m training.plots \
  --pre outputs/eval_pre.json \
  --post outputs/eval_post.json \
  --trainer-state training/sentinel_qwen15_grpo/trainer_state.json \
  --reward-report-task3 outputs/reward_report_task3_seed42.json \
  --cluster-health outputs/cluster_health_history.json \
  --out-dir outputs/charts
