# SENTINEL Visual System

This file is the diagram source of truth. Every diagram used in README, UI, blog, or slides should be derived from here.

## Diagram Inventory

| Diagram | Purpose | Status |
| --- | --- | --- |
| System stack | show the code architecture | ready |
| Episode lifecycle | explain `reset()` to terminal reward | ready |
| Trust and reward flow | show how state turns into learning signal | ready |
| Before / after | show why SENTINEL matters | ready |
| Theme fit | map the project to the hackathon | ready |
| Training loop | show OpenEnv -> TRL / Unsloth pipeline | ready |

---

## 1. System Stack

```mermaid
flowchart TD
  A["HTTP client / UI / inference.py"] --> B["app.py<br/>FastAPI on port 7860"]
  B --> C["SentinelEnv<br/>environment.py"]
  B --> D["_sessions<br/>session_id -> SentinelEnv"]
  C --> E["TaskGraph<br/>task_graph.py"]
  C --> F["TrustLedger<br/>trust_ledger.py"]
  C --> G["SpecialistPool<br/>specialists.py"]
  C --> H["RewardEngine<br/>graders.py"]
  C --> I["Scenario dataset<br/>scenarios.py"]
  C --> J["Typed models<br/>models.py"]
  B --> K["openenv.yaml"]
  B --> L["static/index.html"]
```

---

## 2. Episode Lifecycle

```mermaid
flowchart TD
  A["reset(task_type, seed)"] --> B["sample scenario"]
  B --> C["reshuffle hidden specialist profiles"]
  C --> D["set trust priors to 0.50"]
  D --> E["build task graph"]
  E --> F["return first observation"]

  F --> G["orchestrator chooses action"]
  G --> H["delegate / verify / self solve / skip"]
  H --> I["specialist or self execution"]
  I --> J["record outcome in TaskGraph"]
  J --> K["update TrustLedger"]
  K --> L["compute step reward"]
  L --> M{"done?"}
  M -- "no" --> N["return next observation"]
  N --> G
  M -- "yes" --> O["compute terminal reward"]
  O --> P["return done=True with final info"]
```

---

## 3. Trust And Reward Flow

```mermaid
flowchart LR
  A["Observation<br/>subtask, stakes, trust snapshot"] --> B["Action choice"]
  B --> C["Specialist result<br/>outcome, confidence, adversarial flag, step_cost"]
  C --> D["TaskGraph update"]
  C --> E["TrustLedger Bayesian update"]
  D --> F["completion, detections, poisonings"]
  E --> G["calibration state"]
  F --> H["RewardEngine"]
  G --> H
  H --> I["step reward"]
  H --> J["terminal reward"]
```

---

## 4. Before / After

```mermaid
flowchart LR
  subgraph BEFORE["Before SENTINEL"]
    A1["Uniform trust"] --> A2["Blind delegation"]
    A2 --> A3["Poison accepted at high stakes"]
    A3 --> A4["Downstream subtasks inherit bad state"]
    A4 --> A5["Mission drifts or fails"]
  end

  subgraph AFTER["After SENTINEL"]
    B1["Behavior updates trust"] --> B2["Low-trust high-stakes node detected"]
    B2 --> B3["Verify instead of delegate"]
    B3 --> B4["Poison blocked before cascade"]
    B4 --> B5["Mission completes cleanly"]
  end
```

---

## 5. Theme Fit

```mermaid
flowchart TD
  S["SENTINEL"] --> T1["Theme 1<br/>multi-agent interaction"]
  S --> T2["Theme 2<br/>long-horizon planning"]
  S --> T4["Theme 4<br/>self-improvement"]
  S --> T5["Theme 5<br/>wild card"]

  T1 --> B1["orchestrator + five specialists<br/>partial observability<br/>adversarial dynamics"]
  T2 --> B2["task graph<br/>step budget pressure<br/>delayed terminal reward"]
  T4 --> B3["profile reshuffle<br/>auto-curriculum<br/>no memorization"]
  T5 --> B4["real production weakness<br/>blind trust in agent pipelines"]
```

---

## 6. Training Loop

```mermaid
flowchart LR
  A["Prompt / observation"] --> B["Model rollout"]
  B --> C["Action text or structured action"]
  C --> D["SENTINEL environment"]
  D --> E["Reward + next observation"]
  E --> F["TRL / GRPO trainer"]
  F --> G["updated policy"]
  G --> B

  H["training/evaluate.py"] --> I["random / heuristic / oracle-lite"]
  I --> J["evaluation_results.json"]
  I --> K["baseline_comparison.png"]
```

---

## Use Rules

1. Do not invent new component names in slide decks that do not exist in code.
2. Use `SentinelEnv`, `TrustLedger`, `SpecialistPool`, `TaskGraph`, `RewardEngine` consistently.
3. Use real baseline numbers in public before/after materials.
4. Export polished PNG versions from these mermaid sources later, but keep this file as the editable truth.
