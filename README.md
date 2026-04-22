# SENTINEL

Self-Evolving Network for Training Intelligent Agents Under Adversarial Long-Horizon Tasks.

SENTINEL is an OpenEnv-compatible RL environment for one core skill: training an orchestrator to decide who to trust, when to verify, how to recover, and how to finish long multi-agent work when specialist agents are unreliable or adversarial.

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
curl http://localhost:7860/metadata
curl http://localhost:7860/tasks
curl http://localhost:7860/schema
```

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
python -m py_compile app.py environment.py models.py graders.py specialists.py trust_ledger.py task_graph.py scenarios.py inference.py
python -m pytest -q
python inference.py
python training/evaluate.py --episodes 20 --task task3
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

The evaluator writes `outputs/evaluation_results.json` for demo charts.

## Hackathon Alignment

- Theme 1: multi-agent interaction, partial observability, adversarial specialist, trust calibration.
- Theme 2: long-horizon task graphs with delayed terminal reward and failure recovery.
- Theme 3.1: professional agent orchestration workflow with API-style actions.
- Theme 4: profile shuffle creates a self-resetting curriculum.
- Theme 5: targets a real AI systems failure: blind trust inside agent pipelines.

Winning demo line:

> Agents fail because they trust blindly. SENTINEL trains skepticism, recovery, and oversight.
