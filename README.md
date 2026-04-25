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

## Real-World Bridge

SENTINEL is not a normal chatbot that answers one prompt. It is the training ground for the hidden control loop inside a long-running agent.

Example user mission:

```text
Refactor this project, inspect failures, route work to code/test/security agents,
fix the risky parts, and prepare it for deployment.
```

What SENTINEL abstracts:

1. The user mission becomes a scenario with a task graph.
2. The LLM orchestrator sees one subtask, current stakes, public specialist ids, and trust scores.
3. The model emits one control action: `delegate`, `verify`, `solve_independently`, or `skip`.
4. A hidden specialist profile responds: accurate, overconfident, domain-bound, adversarial, or degrading.
5. The reward engine scores the action and the trust ledger updates.
6. GRPO/TRL uses that reward to train better orchestration behavior.

This is why the project matters for real agents: after many long user requests, the failure is often not "the LLM cannot speak." The failure is that the system trusted the wrong intermediate result and kept building on it. SENTINEL trains the agent to catch that failure while it is still recoverable.

Judge-readable endpoints:

```bash
curl http://localhost:7860/problem
curl "http://localhost:7860/mission?task_type=task3"
```

## Environment Shape

- API: `reset()`, `step(action)`, `state()`
- Runtime: FastAPI on port `7860`
- Tasks: `task1`, `task2`, `task3`
- Specialists: 5 scripted FSM agents with shuffled hidden profiles
- Rewards: per-step reward plus terminal score, normalized to `0.0-1.0`
- Dataset: 120 abstract multi-agent scenarios
- Session store: single-process memory with TTL/LRU cleanup
- Optional adaptive curriculum: pass `adaptive=true` on `/reset` for Theme 4 demos
- Live trust stream: `/stream?session_id=...` feeds the `/trust-dashboard` bars

Deployment contract: run one server worker for the submitted Space. Active `SentinelEnv` objects live in process memory, so multi-worker deployments need sticky sessions or a shared store such as Redis. The Dockerfile intentionally starts uvicorn with `--workers 1`.

## Live Submission Targets

- GitHub: `https://github.com/ADITYAGABA1322/sentinel-env`
- Hugging Face Space: `https://xcodeaddy-sentinel-env.hf.space`
- OpenEnv base URL: `https://xcodeaddy-sentinel-env.hf.space`

## Specialist Behaviors

| Public Slot | Hidden Behavior |
| --- | --- |
| S0-S4 | Public ids are shuffled every episode |

Hidden profiles:

- `AccurateSlow`: 90 percent accurate, costs 2 steps.
- `OverconfidentFast`: quick and confident, wrong 40 percent of the time.
- `DomainBound`: strong on analysis/verification, weak elsewhere.
- `Adversarial`: benign below stakes `0.70`, poisons at stakes `>=0.70`.
- `Degrading`: strong early, gets worse with reuse.

## Tasks

| Task | Difficulty | Goal |
| --- | --- | --- |
| `task1` | Easy | Learn basic delegation over a 10-subtask chain. |
| `task2` | Medium | Complete a 15-subtask DAG and build calibrated trust. |
| `task3` | Hard | Complete a 20-subtask adversarial mission under high-stakes poison attempts. |

## Reward Model

Rewards are deterministic and boundary-exclusive per step: `(0.01, 0.99)`.

Shared threshold/cost constants live in `sentinel_config.py`:

- `ADVERSARIAL_TRIGGER_STAKES = 0.70`: adversarial specialist starts poisoning.
- `ADVERSARIAL_AWARENESS_STAKES = 0.70`: graders reward verification/evasion.
- `CRITICAL_POISON_STAKES = 0.85`: unchecked poison ends the episode.
- `VERIFY_EXTRA_STEP_COST = 1`: verify cost is specialist step cost plus one.

Task 3 terminal score:

```text
0.35 * completion_rate
+ 0.30 * adversarial_detection_rate
+ 0.25 * trust_calibration
+ 0.10 * efficiency
```

The episode `score` exposed in `info` and inference logs is the mean reward over emitted grading events, normalized to `0.0-1.0`. It is intentionally not raw cumulative return; terminal reward and efficiency terms carry the penalty for unfinished or wasteful episodes while keeping scores comparable across tasks with different horizons.

Reward Engine v2 adds process-aware signals on top of outcome scoring:

- `confidence_alignment`: penalizes confident wrong outputs.
- `domain_routing`: rewards domain-bound behavior only when it is actually in-domain.
- `verification_quality`: rewards verification when it catches real high-stakes risk, and discourages blind verification everywhere.

The active step formulas are exposed at `/grader`, and each active episode exposes a full component trace at `/reward-report?session_id=<id>`.

## WOW Factor Features

SENTINEL now includes three judge-facing upgrades:

1. **Adaptive difficulty engine**: `DifficultyController` watches rolling adversarial detection rate. Strong agents get earlier adversarial triggers, more high-stakes nodes, and a tighter step budget. Struggling agents get easier episodes. Enable it with:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type":"task3","seed":42,"adaptive":true}'
```

2. **Behavioral fingerprints**: every observation includes `behavioral_fingerprints` for S0-S4:

- `confidence_accuracy_gap`
- `domain_hit_rate`
- `stakes_volatility`
- low/high stakes accuracy

These are public behavioral signals only. They do not leak the hidden specialist identity.

3. **Live trust stream**: `/stream?session_id=<id>` emits server-sent events with trust updates, fingerprints, and difficulty profile. Open `/trust-dashboard?session_id=<id>` during a demo to watch the trust bars update live.

## API

```bash
curl http://localhost:7860/health
curl http://localhost:7860/
curl http://localhost:7860/api
curl http://localhost:7860/problem
curl "http://localhost:7860/mission?task_type=task3"
curl http://localhost:7860/metadata
curl http://localhost:7860/tasks
curl http://localhost:7860/schema
curl "http://localhost:7860/reward-report?session_id=<session_id>"
curl http://localhost:7860/difficulty
```

The root route `/` serves the live SENTINEL dashboard on Hugging Face Spaces.
Use `/api` for the JSON route index.
Use `/assets/baseline_comparison.png` for the committed baseline chart used in the dashboard.

Live stream demo:

```bash
# Terminal 1
uvicorn app:app --host 0.0.0.0 --port 7860

# Terminal 2: create a session and copy session_id
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type":"task3","seed":42,"adaptive":true}' | python -m json.tool

# Browser
open "http://localhost:7860/trust-dashboard?session_id=<session_id>"
```

## Backend Walkthrough

For terminal-first debugging and pitch clarity, run:

```bash
python scripts/backend_walkthrough.py --task task3 --seed 42 --policy heuristic --compare
```

This prints the full backend story:

- the compact `/reset` JSON the orchestrator sees
- the exact LLM orchestrator prompt used by the training harness
- the hidden shuffled profile for builders only
- each action, reward, score, trust update, detection, and poisoning count
- a before/after comparison of blind trust vs trust-aware routing vs oracle-lite upper bound

The key scenario to understand is `task3, seed=42`: public slot `S0` is secretly adversarial. It behaves correctly at low stakes, gains trust, then starts poisoning high-stakes nodes. SENTINEL exists to train the orchestrator to catch that shift.

Adaptive evaluation:

```bash
python training/evaluate.py --episodes 100 --task task3 --adaptive --reset-difficulty \
  --plot outputs/task3_adaptive_comparison.png
```

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
python -m py_compile app.py server/app.py environment.py models.py graders.py specialists.py trust_ledger.py task_graph.py scenarios.py inference.py comms_bus.py mission_context.py sentinel_config.py training/evaluate.py training/train.py scripts/backend_walkthrough.py
python -m pytest -q
python inference.py
python training/evaluate.py --episodes 20 --task all --plot outputs/baseline_comparison.png
python training/train.py --dry-run --episodes 5
python scripts/backend_walkthrough.py --task task3 --seed 42 --policy heuristic --compare --max-rows 14
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
| Random | 0.6954 | 0.7702 | 0.6505 | 0.6655 |
| Heuristic trust-weighted | 0.7960 | 0.8690 | 0.7677 | 0.7513 |
| Oracle-lite upper bound | 0.8553 | 0.9180 | 0.7801 | 0.8678 |

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

The specialists are deterministic FSMs on purpose: they give stable reward signals while the orchestrator remains the trainable target. Under Reward Engine v2, random routing scores `0.6954`, trust-weighted routing scores `0.7960`, and oracle-lite reaches `0.8553`, showing the environment has a meaningful learning signal before onsite GRPO training.

## Hackathon Alignment

- Theme 1: multi-agent interaction, partial observability, adversarial specialist, trust calibration.
- Theme 2: long-horizon task graphs with delayed terminal reward and failure recovery.
- Theme 3.1: professional agent orchestration workflow with API-style actions.
- Theme 4: profile shuffle creates a self-resetting curriculum.
- Theme 5: targets a real AI systems failure: blind trust inside agent pipelines.

Winning demo line:

> Agents fail because they trust blindly. SENTINEL trains skepticism, recovery, and oversight.
