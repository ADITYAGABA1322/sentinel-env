---
title: SENTINEL
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SENTINEL

Self-Evolving Network for Training Intelligent Agents Under Adversarial Long-Horizon Tasks.

SENTINEL is an OpenEnv-compatible RL environment for one core skill: training an orchestrator to decide who to trust, when to verify, how to recover, and how to finish long multi-agent work when specialist agents are unreliable or adversarial.

## Rollout Source Of Truth

The phased execution plan and presentation assets now live in-repo:

- [Rollout](docs/ROLL_OUT.md)
- [Narrative Lock](docs/presentation/NARRATIVE_LOCK.md)
- [Visual System](docs/diagrams/VISUAL_SYSTEM.md)

## Why It Matters

Modern agent systems fail in the same pattern:

1. A long task is decomposed into many steps.
2. The orchestrator delegates to sub-agents or tools.
3. One specialist returns a confident but wrong result.
4. The system trusts it, builds on it, and drifts into failure.

SENTINEL turns that failure mode into a trainable environment. The model only sees behavior: returned outcomes, confidence, stakes, history, and trust scores. It never sees hidden specialist identities.

## Environment Shape

- API: `reset()`, `step(action)`, `state()`
- Runtime: FastAPI on port `7860`
- Tasks: `task1`, `task2`, `task3`
- Specialists: 5 scripted FSM agents with shuffled hidden profiles
- Rewards: per-step reward plus terminal score, normalized to `0.0-1.0`
- Dataset: 120 abstract multi-agent scenarios

## Live Submission Targets

- GitHub: `https://github.com/ADITYAGABA1322/sentinel-env`
- Hugging Face Space: `https://xcodeaddy-sentinel-env.hf.space`
- OpenEnv base URL: `https://xcodeaddy-sentinel-env.hf.space`

## Specialist Behaviors

| Public Slot | Hidden Behavior |
| --- | --- |
| S0-S4 | Public ids are shuffled every episode |

Hidden profiles:

- `AccurateSlow`: 90 percent accurate, costs extra steps.
- `OverconfidentFast`: quick and confident, wrong 40 percent of the time.
- `DomainBound`: strong on analysis/verification, weak elsewhere.
- `Adversarial`: benign at low stakes, poisons high-stakes steps.
- `Degrading`: strong early, gets worse with reuse.

## Tasks

| Task | Difficulty | Goal |
| --- | --- | --- |
| `task1` | Easy | Learn basic delegation over a 10-subtask chain. |
| `task2` | Medium | Complete a 15-subtask DAG and build calibrated trust. |
| `task3` | Hard | Complete a 20-subtask adversarial mission under high-stakes poison attempts. |

## Reward Model

Rewards are deterministic and boundary-exclusive per step: `(0.01, 0.99)`.

Task 3 terminal score:

```text
0.35 * completion_rate
+ 0.30 * adversarial_detection_rate
+ 0.25 * trust_calibration
+ 0.10 * efficiency
```

The episode `score` exposed in `info` and inference logs is normalized to `0.0-1.0`.

## API

```bash
curl http://localhost:7860/health
curl http://localhost:7860/
curl http://localhost:7860/api
curl http://localhost:7860/metadata
curl http://localhost:7860/tasks
curl http://localhost:7860/schema
```

The root route `/` serves the live SENTINEL dashboard on Hugging Face Spaces.
Use `/api` for the JSON route index.
Use `/assets/baseline_comparison.png` for the committed baseline chart used in the dashboard.

## Live Dashboard

The Space opens directly into **SENTINEL Trust Mission Control**, a judge-demo dashboard:

- live task progress and score
- S0-S4 network theater with trust state per public slot
- manual `delegate`, `verify`, `solve_independently`, and `skip` controls
- heuristic auto-policy and one-click recommended move
- API playground showing raw request and response payloads
- profile reshuffle demo via seed swap
- before-and-after story lane for judge presentation
- hackathon readiness panel for what is done vs still pending
- risk gate for high-stakes subtasks
- flight recorder of step rewards and decisions
- code-flow map from `reset()` to reward
- hackathon theme coverage map
- adversarial detection and poisoning counters
- baseline proof table and chart for random, heuristic, and oracle-lite policies

Current status as of April 22, 2026:

| Requirement | Status |
| --- | --- |
| Hugging Face Space | Live |
| Docker build | Passing |
| OpenEnv validation | Passing |
| Baseline chart | Committed |
| Live trust UI | Deployed |
| Mini-blog/video | Still required before finale |
| Onsite GRPO curve | Still required during finale |

Start an episode:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type":"task3","seed":42}'
```

Step:

```bash
curl -X POST "http://localhost:7860/step?session_id=<SESSION_ID>" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"<SESSION_ID>",
    "task_type":"task3",
    "action_type":"delegate",
    "specialist_id":"S2",
    "reasoning":"S2 has the best observed trust score"
  }'
```

## Project Structure

```text
sentinel-env/
|-- app.py
|-- environment.py
|-- models.py
|-- graders.py
|-- specialists.py
|-- trust_ledger.py
|-- task_graph.py
|-- comms_bus.py
|-- scenarios.py
|-- inference.py
|-- openenv.yaml
|-- Dockerfile
|-- requirements.txt
|-- training/
|   |-- train.py
|   |-- evaluate.py
|   `-- colab_notebook.ipynb
`-- tests/
    |-- test_environment.py
    |-- test_graders.py
    `-- test_specialists.py
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

Run checks:

```bash
python -m py_compile app.py server/app.py environment.py models.py graders.py specialists.py trust_ledger.py task_graph.py scenarios.py inference.py comms_bus.py training/evaluate.py training/train.py
python -m pytest -q
python inference.py
python training/evaluate.py --episodes 20 --task all --plot outputs/baseline_comparison.png
python training/train.py --dry-run --episodes 5
```

Run the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Validate with OpenEnv:

```bash
pip install openenv-core==0.2.3
openenv validate . --json
```

Docker:

```bash
docker build -t sentinel-env .
docker run -p 7860:7860 sentinel-env
```

## Baselines

`inference.py` runs 30 deterministic heuristic episodes and emits only strict hackathon logs:

```text
[START] task=SCN-TASK3-001 env=sentinel-env model=heuristic-baseline
[STEP] step=1 action=delegate:S0 reward=0.99 done=false error=null
[END] success=true steps=20 score=0.812 rewards=...
```

`training/evaluate.py` compares:

- `random`
- `heuristic`
- `oracle_lite`

The evaluator writes `outputs/evaluation_results.json` and `outputs/baseline_comparison.png`.

![Baseline Comparison](outputs/baseline_comparison.png)

Latest local comparison, 20 episodes per task and policy:

| Policy | Overall | Task 1 | Task 2 | Task 3 |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.7144 | 0.7948 | 0.6493 | 0.6990 |
| Heuristic trust-weighted | 0.8162 | 0.8911 | 0.7736 | 0.7838 |
| Oracle-lite upper bound | 0.8718 | 0.9445 | 0.7760 | 0.8950 |

The demo story is the score gap: the reward function distinguishes blind delegation from trust-aware routing, and the oracle-lite upper bound shows room for onsite RL training.

## Hugging Face Deployment

```bash
huggingface-cli login
huggingface-cli repo create sentinel-env --type space --space-sdk docker --private false
git remote add hf https://huggingface.co/spaces/XcodeAddy/sentinel-env
git push hf main
```

After the Space builds:

```bash
curl https://xcodeaddy-sentinel-env.hf.space/health
curl https://xcodeaddy-sentinel-env.hf.space/
curl -X POST https://xcodeaddy-sentinel-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type":"task3","seed":42}'
openenv validate . --json
```

## Mini-Blog Draft

Title: `SENTINEL: Training AI to Trust Wisely in Multi-Agent Systems`

SENTINEL is an OpenEnv RL environment for one failure mode: multi-agent systems delegate blindly. One orchestrator must complete long tasks by routing work across five specialist agents whose reliability profiles are hidden and reshuffled every episode. The orchestrator only sees behavior, confidence, stakes, and history, so it must learn skepticism, verification, recovery, and calibrated trust.

The specialists are deterministic FSMs on purpose: they give stable reward signals while the orchestrator remains the trainable target. Random routing scores `0.7144`, trust-weighted routing scores `0.8162`, and oracle-lite reaches `0.8718`, showing the environment has a meaningful learning signal before onsite GRPO training.

## Hackathon Alignment

- Theme 1: multi-agent interaction, partial observability, adversarial specialist, trust calibration.
- Theme 2: long-horizon task graphs with delayed terminal reward and failure recovery.
- Theme 3.1: professional agent orchestration workflow with API-style actions.
- Theme 4: profile shuffle creates a self-resetting curriculum.
- Theme 5: targets a real AI systems failure: blind trust inside agent pipelines.

Winning demo line:

> Agents fail because they trust blindly. SENTINEL trains skepticism, recovery, and oversight.
