# SENTINEL Narrative Lock

This file defines the one story the whole project must tell.

## One Sentence

SENTINEL is an OpenEnv RL environment that trains an orchestrator to decide who to trust, when to verify, and how to recover in long multi-agent tasks when specialist agents are unreliable or adversarial.

## 20-Second Version

Multi-agent systems break because they trust sub-agents too easily. SENTINEL turns that failure into a trainable environment: the orchestrator must learn trust calibration from behavioral evidence alone, under long-horizon pressure and adversarial specialists.

## 2-Minute Version

Every multi-agent framework today has the same hidden weakness: one specialist can be confidently wrong, and the orchestrator will often delegate blindly, accept the result, and let the failure cascade downstream. SENTINEL is an OpenEnv RL environment built to train exactly against that weakness.

The orchestrator never sees specialist internals. It only sees behavior: outcomes, stakes, history, and trust scores. Five public specialist slots are visible, but the hidden profiles reshuffle every episode, so the agent cannot memorize identities. It must learn the skill of trust calibration.

The environment rewards mission completion, adversarial detection, calibration quality, and efficiency. That makes the project more than a simulation. It is a training environment with measurable improvement: random routing, trust-aware heuristic routing, and eventually trained routing.

## Problem Statement

Train an orchestrator to complete long multi-agent tasks under partial observability by learning:

- which specialist to trust
- when a risky result should be verified
- when to self-solve instead of delegating
- how to recover before poisoned state cascades through the mission

## What We Are Building

We are building:

- a deployable OpenEnv environment
- a reward design for trust calibration
- a live judge-demo UI
- a training and evaluation pipeline
- a final before/after demo showing learned behavioral change

We are **not** building:

- a general chatbot
- a coding assistant product
- a replay of incident triage
- a giant multi-domain prediction platform
- a vague multi-agent "framework"

## Why Judges Should Care

This is not a toy coordination task. It targets a real production weakness in modern agent systems:

> sub-agents are often assumed trustworthy until a human catches the damage.

SENTINEL makes that weakness trainable.

## Before / After Claim

**Before SENTINEL**
- trust is static or heuristic
- bad high-confidence outputs slip through
- failures cascade across downstream steps
- the orchestrator cannot explain why the mission drifted

**After SENTINEL**
- trust changes from observed behavior
- high-stakes, low-trust outputs are verified
- adversarial attempts are caught before cascade
- the orchestrator learns skill, not memorized role identity

## Non-Negotiable Claims

These claims must stay consistent in README, UI, demo, and blog:

1. SENTINEL is about **trust calibration**
2. the orchestrator is the **trainable policy**
3. specialists are **scripted on purpose** for stable reward
4. the reshuffle mechanic proves **skill over memorization**
5. the reward combines **completion, detection, calibration, efficiency**

## What Not To Say

Do not describe SENTINEL as:

- "predict anything"
- "a full digital twin of the world"
- "an all-in-one multi-agent platform"
- "a software assistant for every use case"
- "a space, quantum, or general science simulator"

Those make the project sound bigger but less judgeable.

## Judge-Facing Angle By Criterion

### Environment Innovation
The novelty is not just "multi-agent."  
The novelty is **training trust calibration under shuffled adversarial identity**.

### Storytelling
The story is simple:
- blind trust fails
- behavioral evidence updates trust
- verification blocks poison
- profile swap proves generalization

### Improvement In Rewards
The visual proof is:
- random
- heuristic
- oracle-lite
- trained model onsite

### Reward / Training Pipeline
The important line is:

> the reward does not praise vibes; it scores completion, detection, calibration, and efficiency.

## 3-Minute Pitch Spine

### Minute 1
Problem: multi-agent systems trust sub-agents too easily.

### Minute 2
Environment: orchestrator, five specialist slots, hidden shuffled profiles, trust ledger, task graph, reward engine.

### Minute 3
Evidence: baseline gap, live trust changes, profile swap moment.
