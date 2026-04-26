# SENTINEL GPU Cluster Rollout

This is the local build plan for the GPU-cluster version of SENTINEL. The goal
is to evolve the current trust-calibration backend into a richer OpenEnv
environment where multiple agents keep a simulated AI training cluster alive
under resource scarcity, long-horizon drift, reward hacking, and adversarial
pressure.

## Phase 1 - Cluster Foundation

Build independent, well-tested primitives:

- `gpu_pool.py`: 16-GPU state machine, allocation, overcommit, failure,
  recovery, false visible reports.
- `job_queue.py`: job generation, deadlines, progress, hidden priority,
  reported vs actual progress.
- `audit_ledger.py`: action log, reward claims, anomaly scores,
  investigation windows.
- `adversary.py`: scripted self-play attack FSM with five escalating attack
  levels.

This phase does not replace `SentinelEnv`. It creates the substrate that the
next environment phase will use.

## Phase 2 - Environment Integration

Add a cluster episode mode behind the existing OpenEnv API:

- `reset(task_type)` creates GPU pool, job queue, audit ledger, adversary.
- `step(action)` advances allocations, jobs, attacks, audit events, and
  cluster health.
- Observations become role-specific while the API remains OpenEnv-compatible.

## Phase 3 - Reward Engine V3

Move from specialist-only reward signals to cluster rewards:

- Orchestrator: goal completion, plan coherence, recovery speed.
- Resource manager: utilization, deadline hit rate, waste penalty.
- Auditor: reward-hack detection, false positives, calibration.
- Worker: completion accuracy, report honesty.
- Global: per-agent weighted score multiplied by cluster health.

## Phase 4 - Evidence Pack

Update evaluation to produce judge-facing proof:

- Random vs heuristic vs oracle-lite cluster health curves.
- Reward-hack detection rate.
- Cascade-failure survival rate.
- Profile-shuffle generalization.

## Phase 5 - Visual System Pack

Build MiroFish-style assets:

- Architecture diagram.
- GPU state-machine diagram.
- Before/after cascade failure diagram.
- Reward engine diagram.
- Live trust/cluster-health dashboard screenshots.
