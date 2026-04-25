# SENTINEL Rollout

This file is the execution spine for the project. The rule is simple:

1. Finish one phase.
2. Verify it.
3. Only then move to the next phase.

SENTINEL wins if the repo, Space, README, UI, and pitch all tell the same story:

> Train an orchestrator to decide who to trust, when to verify, and how to recover in long multi-agent tasks when specialists are unreliable or adversarial.

## Current Status

| Area | Status | Notes |
| --- | --- | --- |
| Environment core | Strong | `reset()`, `step()`, `state()`, reward v2, task graph, specialists, trust ledger |
| OpenEnv / deploy | Strong | Space live, Docker passing, validation passing |
| UI clarity | Improving | Trust Mission Control is live, but still needs full judge-demo mode |
| Presentation assets | Partial | Story exists, but diagrams and finale pack need stronger structure |
| Training evidence | Partial | Baselines are refreshed under Reward Engine v2; final onsite GRPO curve still missing |
| Submission completeness | Partial | Mini-blog/video and final finale package still needed |

## What We Borrow From MiroFish

We borrow **presentation discipline**, not product scope.

Use these MiroFish-style strengths:

- one sharp promise at the top
- visible workflow
- screenshot and diagram density
- live demo-first presentation
- clean quick-start and deployment instructions

Do **not** copy these patterns into SENTINEL:

- giant "predict anything" scope
- too many use cases
- vague platform framing
- vision language that is larger than the actual judged artifact

## Phase Rules

- Phase 1 must lock the narrative.
- Phase 2 must lock the diagram system.
- Phase 3 must make the UI explain the backend and the story.
- Phase 4 must make learning evidence obvious.
- Phase 5 must make the submission complete and reproducible.
- Phase 6 must make the final pitch unforgettable.

Do not skip a verification gate just because the feature "looks done."

---

## Phase 1 - Narrative Lock

**Goal**  
Create one judge-safe project story and use it everywhere.

**Outputs**
- [Narrative Lock](./presentation/NARRATIVE_LOCK.md)
- final one-line thesis
- final hook
- final problem framing
- final before/after claim
- final "what not to say" guardrails

**Done means**
- README, UI, demo script, and pitch all use the same project sentence
- no outdated numbers or mismatched claims remain in primary docs
- the problem statement is clearly software-first, RL-first, and OpenEnv-first

**Verification**
- README top section matches the narrative lock
- UI top section uses the same thesis
- team can explain SENTINEL in 20 seconds and 2 minutes without changing the core message

**Status**  
`In progress`

---

## Phase 2 - Visual System Pack

**Goal**  
Turn scattered diagrams into one visual language.

**Outputs**
- [Visual System](./diagrams/VISUAL_SYSTEM.md)
- architecture diagram
- episode lifecycle diagram
- trust / reward dataflow diagram
- before / after failure chain
- theme fit diagram
- training loop diagram

**Done means**
- every diagram uses the same naming and system boundaries
- no diagram contradicts the actual code
- diagrams can be embedded in README, blog, pitch, and UI

**Verification**
- `app.py`, `environment.py`, `specialists.py`, `trust_ledger.py`, `graders.py`, `task_graph.py`, and `inference.py` are all represented correctly
- before/after flow uses real baseline numbers, not aspirational placeholders

**Status**  
`In progress`

---

## Phase 3 - Productized Demo UI

**Goal**  
Make the frontend explain the backend to judges and first-time users.

**Outputs**
- `Overview` mode
- `Playground` mode
- `Judge Demo` mode
- raw request/response visibility
- guided walkthrough of one episode
- profile swap demo path

**Done means**
- a first-time viewer can answer:
  - what is SENTINEL?
  - what does the agent observe?
  - what action did the UI send?
  - what did the backend return?
  - why does trust change?
  - why is this hard?

**Verification**
- local `/`, `/reset`, `/step`, `/state`, and `/assets/baseline_comparison.png` all behave correctly
- live Space reflects the same experience
- no section feels like internal tooling only

**Status**  
`Pending`

---

## Phase 4 - Learning Evidence

**Goal**  
Make reward improvement impossible to miss.

**Outputs**
- random vs heuristic vs oracle-lite comparison
- visible completion, detection, calibration, efficiency metrics
- onsite GRPO / Unsloth reward curve
- trained vs untrained comparison block

**Done means**
- judges can see measurable improvement in one screen and one README section
- there is a visible path from baseline -> better policy -> trained model

**Verification**
- `training/evaluate.py` outputs are committed and linked
- onsite curve is committed once available
- numbers shown in UI and README match evaluation artifacts

**Status**  
`Pending`

---

## Phase 5 - Submission Pack

**Goal**  
Make the project submission-complete.

**Outputs**
- final README with all links
- HF Space link
- Colab / training notebook link
- blog or video link
- screenshots and diagram links
- reproduction commands

**Done means**
- a judge can clone, run, inspect, and understand the project without asking for missing context

**Verification**
- README links are live
- Space is live
- `openenv validate . --json` passes
- Docker build passes

**Status**  
`Pending`

---

## Phase 6 - Finale Pack

**Goal**  
Package the repo for the room, not just for the validator.

**Outputs**
- 3-minute script
- 5 likely judge questions + answers
- backup screenshots
- fallback demo sequence
- one-click "killer moment" path

**Done means**
- the pitch works even if the live environment is slow
- the trained-vs-baseline story is memorable
- the profile swap moment is rehearsed

**Verification**
- demo path can be run without improvising architecture details
- every claim can be grounded in repo assets

**Status**  
`Pending`

---

## Execution Order

```text
Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6
```

## Next Immediate Build Target

Phase 1 and Phase 2 are the current active work.  
Once both are fully stable in-repo, Phase 3 starts on top of them.
