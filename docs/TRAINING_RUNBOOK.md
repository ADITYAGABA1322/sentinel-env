# SENTINEL Training Runbook

This is the exact path for training SENTINEL during the hackathon without
putting GPU work inside the Hugging Face Space runtime.

## Mental Model

SENTINEL is not trained from a normal static CSV of prompt-answer pairs.

The loop is:

```text
reset() observation -> model emits JSON action -> step(action) -> reward -> GRPO update
```

The environment is the dataset generator and the reward engine is the teacher.
The scripted specialists/workers are not trained. The first trained model is the
orchestrator policy that chooses actions.

## Data We Have

Abstract trust environment:

```text
task1: 40 scenarios x 10 subtasks = 400 nodes
task2: 40 scenarios x 15 subtasks = 600 nodes
task3: 40 scenarios x 20 subtasks = 800 nodes
total: 120 scenarios, 1,800 subtask nodes
```

GPU cluster environment:

```text
task1: 10 jobs, 8 GPUs, 30 steps
task2: 20 jobs, 12 GPUs, 60 steps
task3: 30 jobs, 16 GPUs, 120 steps
```

The cluster environment is procedural. Changing the seed creates new job
queues, hidden worker shuffles, attacks, and failure traces.

## SFT vs GRPO

Use SFT when you already have ideal demonstrations:

```text
prompt -> ideal JSON action
```

Use GRPO/RL when you can verify actions programmatically:

```text
prompt -> sampled JSON action -> environment reward
```

For SENTINEL, GRPO is the right headline because the reward is objective:
completion, detection, calibration, efficiency, and anti-hack signals. A small
SFT warmup can be added later by recording heuristic/oracle actions, but it is
not required for the first demo.

## Colab Free T4 Flow

1. Open `training/colab_notebook.ipynb` in Google Colab.
2. Runtime -> Change runtime type -> T4 GPU.
3. Run cells 1-4 to install dependencies and log in to Hugging Face.
4. Run a smoke training with 50-100 episodes.
5. Run the full training with 200 episodes when the smoke run looks good.
6. Generate replay JSONL and charts.
7. Commit `outputs/charts/*.png` and `outputs/trained_policy_replay.jsonl`.

## Why Replay Exists

The live Hugging Face Space should stay cheap and deterministic. It should not
load Qwen or a LoRA adapter at runtime.

After Colab training, the notebook records the trained model's actions:

```json
{"task_type":"task3","seed":42,"step":7,"action":{"action_type":"verify","specialist_id":"S0"}}
```

The Space can replay those actions as a fourth policy called `GRPO`. If the
current seed is missing from the replay table, it falls back to the heuristic
and marks the row as a replay miss.

## Commands

Pre-training baseline:

```bash
python training/evaluate.py --episodes 30 --task all \
  --out outputs/eval_pre.json --no-plot
```

Train:

```bash
python training/train.py \
  --episodes 200 --task all --seed 0 \
  --model unsloth/Qwen2.5-1.5B-Instruct \
  --epochs 1 --batch-size 2 --learning-rate 5e-6 \
  --lora-rank 16 --max-seq-length 1024 \
  --output-dir training/sentinel_qwen15_grpo
```

Record replay:

```python
from training.replay import record_trained_actions

record_trained_actions(
    adapter_path="training/sentinel_qwen15_grpo",
    base_model="unsloth/Qwen2.5-1.5B-Instruct",
    tasks=["task1", "task2", "task3"],
    seeds=range(30),
    out_path="outputs/trained_policy_replay.jsonl",
)
```

Post-training replay eval:

```bash
python training/evaluate.py --episodes 30 --task all \
  --policies random,heuristic,oracle_lite,trained \
  --replay outputs/trained_policy_replay.jsonl \
  --out outputs/eval_post.json --no-plot
```

Generate charts:

```bash
python -m training.plots \
  --pre outputs/eval_pre.json \
  --post outputs/eval_post.json \
  --trainer-state training/sentinel_qwen15_grpo/trainer_state.json \
  --reward-report-task3 outputs/reward_report_task3_seed42.json \
  --cluster-health outputs/cluster_health_history.json \
  --out-dir outputs/charts
```

## Hugging Face Token Usage

Use a Hugging Face token in Colab for:

- downloading gated/private models if needed,
- uploading the LoRA adapter to your namespace,
- pushing final chart/replay artifacts if you commit from Colab.

The Space itself does not need GPU to run the replay demo.

## Hugging Face App URLs

Use these two Hugging Face URLs for different jobs:

```text
https://huggingface.co/spaces/XcodeAddy/sentinel-env
```

This is the Space repository/settings page. Use it to inspect files, Settings,
hardware, build logs, variables, secrets, and commits. It is not the iframe app
URL you demo to judges.

```text
https://xcodeaddy-sentinel-env.hf.space/
```

This is the real live app URL. Use this for the dashboard, API smoke tests, and
OpenEnv base URL.

When running locally, start uvicorn with `--host 0.0.0.0`, but open the browser
at `http://127.0.0.1:7860/` or `http://localhost:7860/`. Do not browse to
`http://0.0.0.0:7860/`; `0.0.0.0` is only a bind address.

## Hugging Face Credits

Best use:

- keep the Space on CPU for normal judging,
- optionally upgrade the Space to T4 only during the final live demo if the UI
  needs extra responsiveness,
- avoid doing full training inside the Space,
- use Hugging Face Jobs or Colab for the actual GRPO run.

The Space is for serving the environment and replay demo. Training belongs in
Colab or in a Hugging Face GPU Job.

HF Jobs smoke path:

```bash
.venv/bin/python training/launch_hf_job.py \
  --mode import-smoke \
  --timeout 45m

.venv/bin/python training/launch_hf_job.py \
  --mode train-smoke \
  --episodes 50 \
  --timeout 2h
```

If `import-smoke` passes, run the full job:

```bash
.venv/bin/python training/launch_hf_job.py \
  --mode train-full \
  --episodes 200 \
  --timeout 4h
```

The launcher uses `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` because the
current Unsloth stack pulls `torchao`, which expects torch `>=2.11`.

## Success Criteria

Before the final demo, make sure these exist:

```text
outputs/trained_policy_replay.jsonl
outputs/charts/baseline_grouped_bars.png
outputs/charts/grpo_reward_curve.png
outputs/charts/trust_evolution.png
outputs/charts/detection_vs_poisoning.png
outputs/charts/cluster_health_timeline.png
outputs/charts/task_radar.png
outputs/charts/ablation.png
outputs/charts/baseline_delta_lines.png
outputs/charts/cluster_health_policy_lines.png
outputs/charts/trust_gap_over_time.png
outputs/charts/reward_component_stacked_area.png
outputs/charts/failure_fishbone_map.png
```

Then verify:

```bash
python -m pytest -q
python training/evaluate.py --episodes 5 --task task3 \
  --policies random,heuristic,oracle_lite,trained \
  --replay outputs/trained_policy_replay.jsonl
```
