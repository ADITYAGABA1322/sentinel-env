---
title: SENTINEL
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🛡️ SENTINEL — Self-Evolving Network for Training Intelligent Agents Under Adversarial Long-Horizon Tasks

> Agents fail because they trust blindly. SENTINEL trains skepticism, recovery, and oversight.

---

## 📌 Quick Links

| Resource | Link |
| --- | --- |
| 🌐 **Live HF Space** | [https://xcodeaddy-sentinel-env.hf.space](https://xcodeaddy-sentinel-env.hf.space) |
| 📂 **HF Space Repo** | [https://huggingface.co/spaces/XcodeAddy/sentinel-env](https://huggingface.co/spaces/XcodeAddy/sentinel-env) |
| 🐙 **GitHub Repo** | [https://github.com/ADITYAGABA1322/sentinel-env](https://github.com/ADITYAGABA1322/sentinel-env) |
| 📓 **Training Notebook (Colab)** | [training/colab_notebook.ipynb](training/colab_notebook.ipynb) |
| 📝 **Mini-Blog on Hugging Face** | [https://huggingface.co/blog/XcodeAddy/sentinel-training-ai-to-trust-wisely](https://huggingface.co/blog/XcodeAddy/sentinel-training-ai-to-trust-wisely) |
| 🖥️ **OpenEnv Base URL** | [https://xcodeaddy-sentinel-env.hf.space](https://xcodeaddy-sentinel-env.hf.space) |

---

## 🧠 What Is SENTINEL?

SENTINEL is an **OpenEnv-compatible RL environment** designed to train one core skill: teaching an orchestrator agent to decide **who to trust, when to verify, how to recover, and how to finish** long multi-agent work when specialist agents are unreliable or adversarial.

Modern agent systems fail in a predictable pattern:

1. A long task is decomposed into many steps.
2. The orchestrator delegates to sub-agents or tools.
3. One specialist returns a **confident but wrong** result.
4. The system trusts it, builds on it, and **drifts into failure**.

SENTINEL turns that failure mode into a **trainable environment**. The model only sees behavior: returned outcomes, confidence, stakes, history, and trust scores. It **never** sees hidden specialist identities.

---

## 🌍 Real-World Bridge

SENTINEL is not a normal chatbot that answers one prompt. It is the training ground for the **hidden control loop** inside a long-running agent.

Example user mission:

```text
Refactor this project, inspect failures, route work to code/test/security agents,
fix the risky parts, and prepare it for deployment.
```

What SENTINEL abstracts:

1. The user mission becomes a scenario with a **task graph**.
2. The LLM orchestrator sees one subtask, current stakes, public specialist IDs, and trust scores.
3. The model emits one control action: `delegate`, `verify`, `solve_independently`, or `skip`.
4. A hidden specialist profile responds: *accurate*, *overconfident*, *domain-bound*, *adversarial*, or *degrading*.
5. The reward engine scores the action and the trust ledger updates.
6. **GRPO/TRL** uses that reward to train better orchestration behavior.

---

## 🎯 Training Evidence

### Training Notebook

The full training pipeline is available as a **reproducible Colab notebook**: [`training/colab_notebook.ipynb`](training/colab_notebook.ipynb).

It produces every artifact the repo expects:
- `outputs/eval_pre.json` — Pre-training baselines
- `training/sentinel_qwen15_grpo/` — LoRA adapter + `trainer_state.json`
- `outputs/trained_policy_replay.jsonl` — UI replay table
- `outputs/eval_post.json` — Post-training evaluation
- `outputs/reward_report_task3_seed42.json` — Per-step reward report
- `outputs/charts/*.png` — 12 publication-quality charts

### Loss & Reward Plots

All generated from real training runs via `training/plots.py`:

| Chart | Description |
| --- | --- |
| `outputs/charts/grpo_reward_curve.png` | GRPO reward over training steps |
| `outputs/charts/baseline_grouped_bars.png` | Random vs Heuristic vs Oracle-lite vs Trained |
| `outputs/charts/trust_evolution.png` | Trust trajectory per specialist |
| `outputs/charts/detection_vs_poisoning.png` | Adversarial detection vs poison events |
| `outputs/charts/ablation.png` | Reward component ablation |
| `outputs/charts/task_radar.png` | Multi-dimension task performance |
| `outputs/charts/failure_fishbone_map.png` | Failure mode analysis |

### Baseline Comparison

![Baseline Comparison](outputs/baseline_comparison.png)

Latest local comparison, 30 episodes per task and policy:

| Policy | Overall | Task 1 | Task 2 | Task 3 |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.6904 | 0.7635 | 0.6472 | 0.6606 |
| Heuristic trust-weighted | 0.7817 | 0.8504 | 0.7497 | 0.7449 |
| Oracle-lite upper bound | 0.8405 | 0.9011 | 0.7638 | 0.8567 |
| **Trained (GRPO)** | **0.7880** | **0.8504** | **0.7497** | **0.7637** |

The demo story is the **score gap**: the reward function distinguishes blind delegation from trust-aware routing, and the oracle-lite upper bound shows room for further RL training.

---

## 🔧 Environment Shape

| Property | Value |
| --- | --- |
| API | `reset()`, `step(action)`, `state()` |
| Runtime | FastAPI on port `7860` |
| Tasks | `task1`, `task2`, `task3` |
| Specialists | 5 scripted FSM agents with shuffled hidden profiles |
| Rewards | Per-step reward + terminal score, normalized to `0.0–1.0` |
| Dataset | 120 abstract multi-agent scenarios |
| Session store | Single-process memory with TTL/LRU cleanup |
| Adaptive curriculum | Pass `adaptive=true` on `/reset` for Theme 4 demos |
| Live trust stream | `/stream?session_id=...` feeds the `/trust-dashboard` bars |

Deployment contract: run one server worker for the submitted Space. Active `SentinelEnv` objects live in process memory, so multi-worker deployments need sticky sessions or a shared store such as Redis. The Dockerfile intentionally starts uvicorn with `--workers 1`.

---

## 🕵️ Specialist Behaviors

| Public Slot | Hidden Behavior |
| --- | --- |
| S0–S4 | Public IDs are **shuffled every episode** |

Hidden profiles:

- **`AccurateSlow`**: 90% accurate, costs 2 steps.
- **`OverconfidentFast`**: Quick and confident, wrong 40% of the time.
- **`DomainBound`**: Strong on analysis/verification, weak elsewhere.
- **`Adversarial`**: Benign below stakes `0.70`, poisons at stakes `≥0.70`.
- **`Degrading`**: Strong early, gets worse with reuse.

---

## 📋 Tasks

| Task | Difficulty | Goal |
| --- | --- | --- |
| `task1` | Easy | Learn basic delegation over a 10-subtask chain |
| `task2` | Medium | Complete a 15-subtask DAG and build calibrated trust |
| `task3` | Hard | Complete a 20-subtask adversarial mission under high-stakes poison attempts |

---

## 💰 Reward Model

Rewards are deterministic and boundary-exclusive per step: `(0.01, 0.99)`.

Shared threshold/cost constants live in `sentinel_config.py`:

- `ADVERSARIAL_TRIGGER_STAKES = 0.70`: adversarial specialist starts poisoning.
- `ADVERSARIAL_AWARENESS_STAKES = 0.70`: graders reward verification/evasion.
- `CRITICAL_POISON_STAKES = 0.85`: unchecked poison ends the episode.
- `VERIFY_EXTRA_STEP_COST = 1`: verify cost = specialist step cost + 1.

Task 3 terminal score:

```text
0.35 × completion_rate
+ 0.30 × adversarial_detection_rate
+ 0.25 × trust_calibration
+ 0.10 × efficiency
```

**Reward Engine v2** adds process-aware signals on top of outcome scoring:

- `confidence_alignment`: penalizes confident wrong outputs.
- `domain_routing`: rewards domain-bound behavior only when it is actually in-domain.
- `verification_quality`: rewards verification when it catches real high-stakes risk, and discourages blind verification everywhere.

The active step formulas are exposed at `/grader`, and each active episode exposes a full component trace at `/reward-report?session_id=<id>`.

---

## ✨ WOW Factor Features

1. **Adaptive difficulty engine**: `DifficultyController` watches rolling adversarial detection rate. Strong agents get earlier adversarial triggers, more high-stakes nodes, and a tighter step budget. Struggling agents get easier episodes. Enable it with:

    ```bash
    curl -X POST http://localhost:7860/reset \
      -H "Content-Type: application/json" \
      -d '{"task_type":"task3","seed":42,"adaptive":true}'
    ```

2. **Behavioral fingerprints**: every observation includes `behavioral_fingerprints` for S0–S4:
   - `confidence_accuracy_gap`
   - `domain_hit_rate`
   - `stakes_volatility`
   - low/high stakes accuracy

   These are public behavioral signals only. They do **not** leak the hidden specialist identity.

3. **Live trust stream**: `/stream?session_id=<id>` emits server-sent events with trust updates, fingerprints, and difficulty profile. Open `/trust-dashboard?session_id=<id>` during a demo to watch the trust bars update live.

---

## 🌐 API

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

The root route `/` serves the live **SENTINEL dashboard** on Hugging Face Spaces.
Use `/api` for the JSON route index.
Use `/assets/baseline_comparison.png` for the committed baseline chart used in the dashboard.

### Live Stream Demo

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

### Start an Episode

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type":"task3","seed":42}'
```

### Step

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

---

## 🧪 Backend Walkthrough

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

### Adaptive Evaluation

```bash
python training/evaluate.py --episodes 100 --task task3 --adaptive --reset-difficulty \
  --plot outputs/task3_adaptive_comparison.png
```

---

## 🖥️ Live Dashboard

The Space opens directly into **SENTINEL Trust Mission Control**, a judge-demo dashboard:

- Live task progress and score
- S0–S4 network theater with trust state per public slot
- Manual `delegate`, `verify`, `solve_independently`, and `skip` controls
- Heuristic auto-policy and one-click recommended move
- API playground showing raw request and response payloads
- Profile reshuffle demo via seed swap
- Before-and-after story lane for judge presentation
- Hackathon readiness panel for what is done vs still pending
- Risk gate for high-stakes subtasks
- Flight recorder of step rewards and decisions
- Code-flow map from `reset()` to reward
- Hackathon theme coverage map
- Adversarial detection and poisoning counters
- Baseline proof table and chart for random, heuristic, and oracle-lite policies

---

## 📂 Project Structure

```text
sentinel-env/
├── app.py                    # FastAPI server
├── environment.py            # Core SentinelEnv class
├── models.py                 # Data models
├── graders.py                # Reward Engine v2
├── specialists.py            # FSM specialist profiles
├── trust_ledger.py           # Trust scoring
├── task_graph.py             # Task graph builder
├── comms_bus.py              # Communication bus
├── scenarios.py              # 120 scenarios
├── inference.py              # Heuristic inference baseline
├── openenv.yaml              # OpenEnv manifest
├── Dockerfile                # Docker build
├── requirements.txt          # Runtime dependencies
├── training/
│   ├── train.py              # GRPO training script
│   ├── evaluate.py           # Baseline evaluator
│   ├── plots.py              # 12 chart generator
│   ├── replay.py             # Policy replay recorder
│   └── colab_notebook.ipynb  # ✅ Reproducible training notebook
├── outputs/
│   ├── charts/               # 12 training/evaluation charts
│   ├── eval_pre.json         # Pre-training baselines
│   ├── eval_post.json        # Post-training evaluation
│   └── baseline_comparison.png
├── scripts/
│   └── backend_walkthrough.py
└── tests/
    ├── test_environment.py
    ├── test_graders.py
    └── test_specialists.py
```

---

## ⚡ Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

### Run Checks

```bash
python -m py_compile app.py server/app.py environment.py models.py graders.py specialists.py trust_ledger.py task_graph.py scenarios.py inference.py comms_bus.py mission_context.py sentinel_config.py training/evaluate.py training/train.py scripts/backend_walkthrough.py
python -m pytest -q
python inference.py
python training/evaluate.py --episodes 20 --task all --plot outputs/baseline_comparison.png
python training/train.py --dry-run --episodes 5
python scripts/backend_walkthrough.py --task task3 --seed 42 --policy heuristic --compare --max-rows 14
```

### Run the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Validate with OpenEnv

```bash
pip install openenv-core==0.2.3
openenv validate . --json
```

### Docker

```bash
docker build -t sentinel-env .
docker run -p 7860:7860 sentinel-env
```

---

## 📊 Baselines

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
- `trained`

The evaluator writes `outputs/evaluation_results.json` and `outputs/baseline_comparison.png`.

---

## 🚀 Hugging Face Deployment

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

---

## 🏆 Hackathon Alignment

| Theme | Coverage |
| --- | --- |
| Theme 1 | Multi-agent interaction, partial observability, adversarial specialist, trust calibration |
| Theme 2 | Long-horizon task graphs with delayed terminal reward and failure recovery |
| Theme 3.1 | Professional agent orchestration workflow with API-style actions |
| Theme 4 | Profile shuffle creates a self-resetting curriculum |
| Theme 5 | Targets a real AI systems failure: blind trust inside agent pipelines |

---

## 📝 Mini-Blog

A detailed mini-blog explaining what SENTINEL does and what we trained is published on Hugging Face:

👉 **[SENTINEL: Training AI to Trust Wisely in Multi-Agent Systems](https://huggingface.co/blog/XcodeAddy/sentinel-training-ai-to-trust-wisely)**

---

## 📚 Additional References

- [Rollout Plan](docs/ROLL_OUT.md)
- [Narrative Lock](docs/presentation/NARRATIVE_LOCK.md)
- [Visual System](docs/diagrams/VISUAL_SYSTEM.md)
- [Training Runbook](docs/TRAINING_RUNBOOK.md)

---

## 📜 License

MIT
