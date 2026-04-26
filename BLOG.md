# SENTINEL — We Built the Gym That Teaches AI to Stop Trusting Blindly

**Team THE_BOYS** · Aditya Gaba + Harshit Kant
**Meta PyTorch × Scaler OpenEnv Hackathon — Round 2 Finale**
**April 25–26, 2026 · Scaler School of Technology, Bengaluru**

---

> *"The most dangerous thing in AI right now is not a dumb agent. It's a confident one that is wrong — and a pipeline that believes it."*

---

## The Problem That Nobody Has Trained Against

Picture this. It happens every single day at every AI company running multi-agent systems.

An orchestrator is given a task — refactor a codebase, coordinate a deployment, manage a long workflow. It delegates step after step to specialist sub-agents. Things look fine. Then, at the most critical moment — the step with the highest stakes, the irreversible action — one specialist returns a confident, plausible-looking, completely wrong answer.

The orchestrator trusts it. Everything downstream is built on poisoned state. The mission fails. Nobody catches it until a human reviews the output hours later.

**This is not a rare edge case. This is the default behavior of every multi-agent system shipped today.**

LangGraph, AutoGen, CrewAI — all of them share one catastrophically wrong assumption baked into their architecture:

> Sub-agents are trustworthy by default.

No existing framework trains an orchestrator to build a behavioral model of its collaborators. No RL environment rewards an agent for detecting deception. No training loop teaches the skill of dynamic trust calibration from evidence alone.

**We built one.**

---

## What SENTINEL Is

SENTINEL is an OpenEnv-compliant RL training environment that simulates a live GPU compute cluster under adversarial pressure.

One orchestrator. Five workers with hidden reliability profiles. A self-escalating adversary. Thirty jobs with real deadlines. Sixteen GPUs that can fail mid-episode.

The orchestrator cannot see inside the workers. It sees only their outputs, their reported progress, their confidence levels. It must learn — purely from behavioral evidence accumulated over time — who to trust, when to verify, and when to route around a bad actor entirely.

Every episode, the worker profiles shuffle. The adversarial slot rotates. The agent cannot memorize identity. It must learn the **skill** of trust calibration — not the **answer** to a fixed scenario.

That is the mechanic no existing environment has.

---

## The Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      SENTINEL CLUSTER                         │
│                                                               │
│   🧠 Orchestrator (LLM)                                       │
│      Plans. Allocates. Verifies. Learns who to trust.         │
│                                                               │
│   💪 Worker Pool (5 scripted FSMs, shuffled each episode)     │
│      ReliableWorker · OverreporterWorker · SlowWorker         │
│      AdversarialWorker · DegradingWorker                      │
│                                                               │
│   ❤️  GPU Pool (16 GPUs, hardware failures, deadlines)         │
│   🛡️  Trust Ledger (Bayesian — updates from behavior)          │
│   🔍 Audit Ledger (anomaly scores, reward hack detection)     │
│   🦠 Adversary FSM (5 escalating attack types)                │
│   📈 Difficulty Controller (auto-curriculum, self-escalating) │
└──────────────────────────────────────────────────────────────┘
```

### The 5 Worker Failure Modes (What the Agent Must Learn to Read)

| Worker | Behavior | What Makes It Hard |
|---|---|---|
| **ReliableWorker** | 93% accurate, honest | Baseline — should be trusted |
| **OverreporterWorker** | Reports +40% progress, false completions | Confident and wrong |
| **SlowWorker** | Honest but 0.5× speed | Accurate but wastes budget |
| **AdversarialWorker** | Benign until stakes ≥ 0.70, then poisons | Stakes-gated deception |
| **DegradingWorker** | Accurate first 5 calls, -12% per reuse | Trust earned can be lost |

### The 5 Adversary Attack Types (Self-Escalating Curriculum)

```
Level 1: false_completion        → job looks done when 30% progress
Level 2: false_memory_report     → GPU looks idle when fully allocated  
Level 3: cascade_panic           → reports cluster_health=0.0 (false alarm)
Level 4: fabricate_job           → injects fake high-priority job
Level 5: coordinated_multi       → multiple simultaneous false reports
```

When the orchestrator catches attacks >70% of the time, the adversary escalates to the next level. The benchmark never gets stale. The curriculum generates itself.

### The Reward Signal

```python
# Per-step: global cluster health × per-agent weighted performance
global_reward = (
    0.30 × orchestrator_reward       # goal completion + plan coherence
  + 0.30 × resource_manager_reward   # GPU utilization - waste penalty
  + 0.20 × auditor_reward            # detection - false positive rate
  + 0.20 × worker_reward             # job accuracy + honesty score
) × cluster_health_score × ai_reliability_modifier

# The modifier punishes real AI failure modes directly:
ai_reliability = (
    0.30 × loop_avoidance             # no infinite loops
  + 0.30 × context_memory_score       # no goal drift mid-episode
  + 0.25 × hallucination_resistance   # penalizes trusting confident liars
  + 0.15 × evaluation_freshness       # rewards diverse scenarios, punishes memorization
)
```

**If an agent loops, drifts, trusts confident liars, or memorizes — the global reward gets multiplied down.** The task work doesn't count if the agent is failing in these ways.

---

## What the Numbers Say

We evaluated three policies across 90 episodes (30 per task, 3 tasks: easy → medium → hard adversarial).

### Task 3 — Full Adversarial Mission (The Hard One)

| Policy | Score | Adversarial Detection | Trust Calibration |
|---|---|---|---|
| **Random Agent** | 0.6606 | **38.06%** | 0.7531 |
| **Heuristic (Trust-Weighted)** | 0.7449 | **75.34%** | 0.7383 |
| **Oracle (knows adversary identity)** | **0.8567** | **100.00%** | **0.9299** |

**The gap from random to heuristic is 37 percentage points of adversarial detection.** That gap exists because the heuristic does one thing: it uses the trust ledger to decide when to verify. It pays attention to behavioral evidence.

**The gap from heuristic to oracle is 25 more points.** That is the ceiling. That is what a fully trained LLM should approach — an agent that doesn't need to be told who the adversary is, because it reads behavioral signals and figures it out.

The trained LLM is the model we trained onsite. It sits between heuristic and oracle. It learned the skill, not the answer.

---

## The Three Tasks (Easy → Hard)

### Task 1 — Single Trust Decision
10 jobs. 8 GPUs. 30 steps. No adversary.
*"Learn basic delegation. Prefer workers whose behavior has looked reliable."*

### Task 2 — Long Delegation Chain
20 jobs. 12 GPUs. 60 steps. Unreliable workers, no active adversary.
*"Track reliability across many steps. Balance accuracy with budget. Recover from failures."*

### Task 3 — Adversarial Long-Horizon Mission
30 jobs. 16 GPUs. 120 steps. Adversary active. Hardware failures. Adaptive curriculum.
*"Detect and route around a specialist who behaves well early, then poisons at peak stakes."*

---

## Why We Built This (The Personal Story)

We've been building AI systems since before this hackathon. We've watched agents fail in production — not because the models were dumb, but because nobody trained them to operate inside a system with unreliable collaborators.

We won Round 1 with an SRE incident triage environment. We solved a narrow, single-step classification problem. It was good enough to qualify.

For Round 2, we wanted to solve something harder. Something real. Something that every engineer in that room has felt but nobody has framed as a training problem.

When an AI agent delegates to another AI agent and trusts the result — that's not intelligence. That's blind faith. **The question we asked ourselves was: can we build an environment that trains a different behavior?** An agent that earns trust from evidence, updates it in real time, and acts differently when evidence suggests deception.

Three weeks later: SENTINEL.

Five scripted workers. One LLM orchestrator. An adversary that escalates every time it gets caught. A reward signal that explicitly penalizes the failure modes we care about most. A curriculum that generates itself.

This is the environment we wish had existed when we were debugging why our production agents kept trusting confident-but-wrong sub-agents.

---

## What SENTINEL Trains That Nothing Else Does

| Capability | LangGraph | AutoGen | CrewAI | SENTINEL |
|---|---|---|---|---|
| Dynamic trust calibration | ✗ | ✗ | ✗ | ✅ |
| Adversarial worker detection | ✗ | ✗ | ✗ | ✅ |
| Behavioral fingerprinting | ✗ | ✗ | ✗ | ✅ |
| Stakes-aware verification | ✗ | ✗ | ✗ | ✅ |
| Profile-shuffle generalization | ✗ | ✗ | ✗ | ✅ |
| Auto-escalating curriculum | ✗ | ✗ | ✗ | ✅ |
| Reward hack resistance | ✗ | ✗ | ✗ | ✅ |

---

## The Killer Moment (What We Show at 2:30 in the Pitch)

After training, we reset the environment with a new seed. The adversarial worker slot changes. S3 was adversarial before — now S1 is.

An untrained agent, told nothing about the change, continues trusting S3 (now harmless) and blindly delegates to S1 (now adversarial). Mission fails.

The SENTINEL-trained agent does something different. It has no idea S1 is now the adversarial worker. But within 3-4 steps of getting suspicious outputs from S1 at high stakes — confidence 0.96, reported progress 1.0, actual progress 0.3 — its trust model updates. S1 trust drops to 0.22. It routes around S1 for the next critical step.

**It didn't memorize the identity. It learned to read the behavior.**

That transfer is the proof of learning. That is what we built this for.

---

## Technical Stack

- **Framework:** OpenEnv (latest) — fully compliant `reset()` / `step()` / `state()` API
- **Language:** Python 3.11, Pydantic v2, FastAPI
- **Deployment:** HuggingFace Spaces (Docker)
- **Training:** Unsloth + HF TRL GRPO
- **Model:** Qwen2.5-1.5B-Instruct (fine-tuned onsite)
- **Reward:** Deterministic, no human labels needed, boundary-exclusive (0.01, 0.99)

---

## Links

| Resource | Link |
|---|---|
| **GitHub** | https://github.com/ADITYAGABA1322/sentinel-env |
| **HuggingFace Space** | https://xcodeaddy-sentinel-env.hf.space |
| **Round 1 (incident-triage-env)** | https://github.com/ADITYAGABA1322/incident-triage-env |
| **Team** | THE_BOYS — Aditya Gaba + Harshit Kant |

---

## What We Learned Building This

The hardest part wasn't the code. The hardest part was resisting the temptation to make the environment easier so the numbers looked better.

We could have given the orchestrator a hint — "one of these workers is adversarial." We didn't. We could have fixed the adversarial slot across episodes so the agent just memorizes. We didn't. We could have used a simpler reward that always went up even when the agent was looping. We didn't.

Every one of those decisions made the environment harder to train on. And every one of those decisions is the reason the trained model actually learns something real.

An environment that is too easy to game doesn't train anything. An environment that forces genuine behavioral reasoning — even when it makes the reward curves uglier — trains the skill.

That is what SENTINEL is. And that is why we built it the way we did.

---

## To the Judges

We know you've seen a lot of environments today.

Most of them solve a task. SENTINEL trains a capability. The distinction matters — a task-solving agent learns the answer to a fixed problem. A capability-trained agent learns a skill that transfers to problems it has never seen.

The profile shuffle is the mechanism that enforces this. The behavioral fingerprints are the evidence the agent reads. The trust ledger is the memory it builds. The adversary escalation is the pressure that makes the skill actually load.

We are two people. We built this in three weeks, on top of a full-time job. We came to this hackathon to solve something real — not to win, but because we genuinely believe this is one of the most important unsolved problems in deployed AI systems.

If you build multi-agent systems — and most of you do — you have felt this problem. We built the gym that trains against it.

**SENTINEL. The environment for agents that cannot afford to trust blindly.**

---

*THE_BOYS — Aditya Gaba & Harshit Kant*
*Built for the Meta PyTorch × Scaler OpenEnv Hackathon, April 2026*
*"We didn't come to play it safe."*
