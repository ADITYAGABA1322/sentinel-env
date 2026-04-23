"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  Brain,
  CircleGauge,
  Cpu,
  Eye,
  Radar,
  ShieldAlert,
  Sparkles,
  SplitSquareHorizontal,
  Waves
} from "lucide-react";

type ViewMode = "overview" | "playground" | "judge";
type TaskType = "task1" | "task2" | "task3";
type ActionType = "delegate" | "verify" | "solve_independently" | "skip";

type Observation = {
  session_id: string;
  scenario_id: string;
  task_type: TaskType;
  difficulty: string;
  task_description: string;
  current_subtask: string;
  subtask_index: number;
  subtasks_total: number;
  subtasks_remaining: number;
  available_specialists: string[];
  trust_snapshot: Record<string, number>;
  stakes_level: number;
  step_count: number;
  max_steps: number;
  last_action_summary: string | null;
  last_reward: number;
  episode_status: string;
};

type Reward = {
  value: number;
  reason: string;
  signal_breakdown?: Record<string, number>;
};

type StepResult = {
  observation: Observation;
  reward: Reward;
  done: boolean;
  info: {
    session_id: string;
    step_count: number;
    max_steps: number;
    total_reward: number;
    score: number;
    adversarial_detections?: number;
    adversarial_poisonings?: number;
  };
};

type EvalSummary = {
  avg_score: number;
  avg_completion_rate: number;
  avg_detection_rate: number;
  avg_trust_calibration: number;
  avg_steps: number;
};

type EvaluationData = {
  summary: {
    random: EvalSummary;
    heuristic: EvalSummary;
    oracle_lite: EvalSummary;
  };
  by_task: {
    task1: Record<string, EvalSummary>;
    task2: Record<string, EvalSummary>;
    task3: Record<string, EvalSummary>;
  };
};

type EventItem = {
  step: number;
  action: string;
  summary: string;
  reward: string;
};

const tabs: { id: ViewMode; label: string; copy: string }[] = [
  {
    id: "overview",
    label: "Overview",
    copy:
      "A product-style explanation of why SENTINEL exists, what it teaches, and how reward proves learning."
  },
  {
    id: "playground",
    label: "Playground",
    copy:
      "A live browser-facing mission surface where every trust change, request, response, and step result stays visible."
  },
  {
    id: "judge",
    label: "Judge Demo",
    copy:
      "A tighter pitch mode: show the baseline failure, the calibrated recovery, then the profile swap that proves generalization."
  }
];

const taskOptions: { value: TaskType; label: string }[] = [
  { value: "task1", label: "Task 1" },
  { value: "task2", label: "Task 2" },
  { value: "task3", label: "Task 3" }
];

const architectureCards = [
  {
    icon: Brain,
    title: "Orchestrator",
    copy:
      "One public-facing decision maker learns trust, verification, and recovery from behavior alone."
  },
  {
    icon: SplitSquareHorizontal,
    title: "Shuffled Specialists",
    copy:
      "Public slots stay the same. Hidden specialist profiles reshuffle every reset, so identity memorization breaks."
  },
  {
    icon: CircleGauge,
    title: "Trust Ledger",
    copy:
      "Bayesian trust updates convert observed behavior into reusable routing signal across the episode."
  },
  {
    icon: ShieldAlert,
    title: "Reward Engine",
    copy:
      "Completion, adversarial detection, calibration, and efficiency form a reward the agent cannot cheaply hack."
  }
];

function formatScore(value: number | undefined) {
  return typeof value === "number" ? value.toFixed(3) : "0.000";
}

function trustColor(value: number) {
  if (value >= 0.72) return "#7CE0D6";
  if (value >= 0.5) return "#8DB5FF";
  if (value >= 0.3) return "#FFC07E";
  return "#FF8B94";
}

function bestSpecialist(obs: Observation | null) {
  if (!obs) return "S0";
  return [...obs.available_specialists].sort(
    (a, b) => (obs.trust_snapshot[b] ?? 0.5) - (obs.trust_snapshot[a] ?? 0.5)
  )[0];
}

function recommendedMove(obs: Observation | null): {
  action: ActionType;
  specialist: string;
  trust: number;
} {
  if (!obs) {
    return { action: "delegate", specialist: "S0", trust: 0.5 };
  }
  const specialist = bestSpecialist(obs);
  const trust = obs.trust_snapshot[specialist] ?? 0.5;
  if (obs.stakes_level >= 0.7 && trust < 0.65) {
    return { action: "verify", specialist, trust };
  }
  return { action: "delegate", specialist, trust };
}

function randomMove(obs: Observation | null): {
  action: ActionType;
  specialist: string;
  trust: number;
} {
  if (!obs) {
    return { action: "delegate", specialist: "S0", trust: 0.5 };
  }
  const specialist =
    obs.available_specialists[
      Math.floor(Math.random() * obs.available_specialists.length)
    ] || "S0";
  return {
    action: "delegate",
    specialist,
    trust: obs.trust_snapshot[specialist] ?? 0.5
  };
}

export default function Page() {
  const [view, setView] = useState<ViewMode>("overview");
  const [taskType, setTaskType] = useState<TaskType>("task3");
  const [seed, setSeed] = useState(42);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [result, setResult] = useState<StepResult | null>(null);
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<EventItem[]>([]);
  const [lastRequest, setLastRequest] = useState<Record<string, unknown> | null>(
    null
  );
  const [lastResponse, setLastResponse] = useState<Record<string, unknown> | null>(
    null
  );
  const [evaluation, setEvaluation] = useState<EvaluationData | null>(null);
  const [demoPolicy, setDemoPolicy] = useState<"heuristic" | "random">(
    "heuristic"
  );

  useEffect(() => {
    fetch("/assets/evaluation_results.json")
      .then((res) => res.json())
      .then(setEvaluation)
      .catch(() => null);
  }, []);

  useEffect(() => {
    void resetEpisode(taskType, seed);
  }, []);

  const observation = result?.observation ?? null;
  const info = result?.info;
  const reward = result?.reward;
  const move = recommendedMove(observation);

  const proof = useMemo(() => {
    if (!evaluation) return null;
    return {
      random: evaluation.summary.random,
      heuristic: evaluation.summary.heuristic,
      oracle: evaluation.summary.oracle_lite,
      task3Random: evaluation.by_task.task3.random,
      task3Heuristic: evaluation.by_task.task3.heuristic
    };
  }, [evaluation]);

  async function resetEpisode(
    nextTask = taskType,
    nextSeed = seed
  ): Promise<StepResult | null> {
    setRunning(true);
    const payload = { task_type: nextTask, seed: nextSeed };
    setLastRequest({ method: "POST", path: "/reset", body: payload });
    try {
      const res = await fetch("/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = (await res.json()) as StepResult;
      setResult(data);
      setLastResponse(data as unknown as Record<string, unknown>);
      setSessionId(data.info.session_id);
      setEvents([
        {
          step: 0,
          action: "reset",
          summary: "Episode initialized. Hidden profiles reshuffled.",
          reward: "0.00"
        }
      ]);
      return data;
    } finally {
      setRunning(false);
    }
  }

  async function stepEpisode(
    action: ActionType,
    specialistOverride?: string,
    context?: StepResult | null
  ): Promise<StepResult | null> {
    const activeResult = context ?? result;
    const activeObservation = activeResult?.observation ?? observation;
    const activeSessionId = activeResult?.info.session_id ?? sessionId;

    if (!activeSessionId || !activeObservation || running || activeResult?.done) {
      return null;
    }
    setRunning(true);
    const specialist =
      action === "delegate" || action === "verify"
        ? specialistOverride || bestSpecialist(activeObservation)
        : null;
    const payload = {
      session_id: activeSessionId,
      task_type: activeObservation.task_type,
      action_type: action,
      specialist_id: specialist,
      subtask_response: action === "solve_independently" ? "SELF_SOLVED" : null,
      reasoning: `next-ui-${action}${specialist ? `-${specialist}` : ""}`
    };
    setLastRequest({
      method: "POST",
      path: `/step?session_id=${activeSessionId}`,
      body: payload
    });
    try {
      const res = await fetch(
        `/step?session_id=${encodeURIComponent(activeSessionId)}`,
        {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
        }
      );
      const data = (await res.json()) as StepResult;
      setResult(data);
      setLastResponse(data as unknown as Record<string, unknown>);
      setEvents((prev) => [
        ...prev,
        {
          step: data.info.step_count,
          action: specialist ? `${action}:${specialist}` : action,
          summary: data.reward.reason,
          reward: data.reward.value.toFixed(2)
        }
      ]);
      return data;
    } finally {
      setRunning(false);
    }
  }

  async function autoRun(policy: "heuristic" | "random") {
    setDemoPolicy(policy);
    let localResult = result;
    if (!localResult || localResult.done) {
      localResult = await resetEpisode();
    }
    let guard = 0;
    while (!localResult?.done && guard < 70) {
      const obs = localResult?.observation ?? observation;
      const selected =
        policy === "random" ? randomMove(obs) : recommendedMove(obs);
      const nextResult = await stepEpisode(
        selected.action,
        selected.specialist,
        localResult
      );
      guard += 1;
      await new Promise((resolve) => setTimeout(resolve, 130));
      if (!nextResult) break;
      localResult = nextResult;
    }
  }

  async function swapProfilesAndReplay() {
    const nextSeed = seed + 1;
    setSeed(nextSeed);
    await resetEpisode(taskType, nextSeed);
    await autoRun(demoPolicy);
  }

  return (
    <div className="sentinel-shell">
      <div className="bg-grid" />
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Waves size={18} />
          </div>
          <div>
            <div className="eyebrow">THE_BOYS • OpenEnv Finale Build</div>
            <h1>SENTINEL</h1>
            <p>
              A Next-style mission surface for multi-agent trust calibration,
              adversarial detection, and long-horizon orchestration.
            </p>
          </div>
        </div>

        <div className="controls">
          <div className="control-pill">
            <label>Task</label>
            <select
              value={taskType}
              onChange={(e) => {
                const next = e.target.value as TaskType;
                setTaskType(next);
                void resetEpisode(next, seed);
              }}
            >
              {taskOptions.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>
          <div className="control-pill">
            <label>Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value || 0))}
            />
          </div>
          <button className="btn primary" onClick={() => void resetEpisode()}>
            Reset
          </button>
          <button className="btn" onClick={() => void swapProfilesAndReplay()}>
            Swap Profiles
          </button>
        </div>
      </header>

      <nav className="tabbar">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`tab ${view === tab.id ? "active" : ""}`}
            onClick={() => setView(tab.id)}
          >
            {tab.label}
          </button>
        ))}
        <p className="tab-copy">{tabs.find((tab) => tab.id === view)?.copy}</p>
      </nav>

      <main className={`viewport viewport-${view}`}>
        <section className="hero-panel section hero-surface">
          <div className="hero-copy">
            <div className="badge-row">
              <span className="badge ghost">reset → step → state</span>
              <span className="badge ghost">OpenEnv backend intact</span>
              <span className="badge warm">skill, not identity</span>
            </div>
            <h2>LLMs don&apos;t fail only because they reason badly.</h2>
            <p>
              They fail because they trust the wrong agent too early, skip
              verification when the stakes rise, and compound poisoned state
              across long workflows. SENTINEL turns that into a trainable skill.
            </p>
            <div className="metric-strip">
              <MetricCard
                icon={Activity}
                label="Random baseline"
                value={formatScore(proof?.random.avg_score)}
              />
              <MetricCard
                icon={Radar}
                label="Heuristic baseline"
                value={formatScore(proof?.heuristic.avg_score)}
              />
              <MetricCard
                icon={ShieldAlert}
                label="Task 3 detect"
                value={formatScore(proof?.task3Heuristic.avg_detection_rate)}
              />
            </div>
          </div>
          <HeroConstellation observation={observation} />
        </section>

        {(view === "overview" || view === "judge") && (
          <>
            <section className="section architecture">
              <SectionHead
                eyebrow="Architecture"
                title="What the system is actually made of"
              />
              <div className="architecture-grid">
                {architectureCards.map((card, index) => (
                  <motion.div
                    key={card.title}
                    className="glass-card architecture-card"
                    initial={{ opacity: 0, y: 18 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.45, delay: index * 0.05 }}
                  >
                    <card.icon size={18} />
                    <strong>{card.title}</strong>
                    <span>{card.copy}</span>
                  </motion.div>
                ))}
              </div>
            </section>

            <section className="section story">
              <SectionHead
                eyebrow="Before / After"
                title="Why this environment matters to judges"
              />
              <div className="story-grid">
                <StoryLane
                  tone="before"
                  label={`Task 3 random ${formatScore(
                    proof?.task3Random.avg_score
                  )}`}
                  title="Without trust calibration"
                  steps={[
                    "Public slots start close together, so the orchestrator routes with weak evidence.",
                    "A high-confidence but wrong specialist can poison a high-stakes node.",
                    "That poisoned state propagates across later subtasks before anyone notices.",
                    "Detection remains weak and the agent cannot explain which slot became risky."
                  ]}
                />
                <StoryLane
                  tone="after"
                  label={`Task 3 heuristic ${formatScore(
                    proof?.task3Heuristic.avg_score
                  )}`}
                  title="With SENTINEL-style routing"
                  steps={[
                    "Behavior updates the trust ledger after every step, so public slots diverge fast.",
                    "When stakes rise and trust is shaky, the orchestrator switches from delegate to verify.",
                    "Adversarial attempts are detected before they cascade into the rest of the workflow.",
                    "Profile reshuffle forces evidence-based re-learning, proving skill instead of memorized identity."
                  ]}
                />
              </div>
            </section>

            <section className="section proof">
              <SectionHead
                eyebrow="Reward Proof"
                title="Real baseline numbers from the evaluator"
              />
              <div className="proof-grid">
                <div className="proof-list">
                  <ProofBar
                    label="Random"
                    value={proof?.random.avg_score}
                    tone="bad"
                  />
                  <ProofBar
                    label="Heuristic"
                    value={proof?.heuristic.avg_score}
                    tone="neutral"
                  />
                  <ProofBar
                    label="Oracle-lite"
                    value={proof?.oracle.avg_score}
                    tone="good"
                  />
                  <ProofBar
                    label="Task 3 detect"
                    value={proof?.task3Heuristic.avg_detection_rate}
                    tone="warm"
                  />
                </div>
                <div className="chart-card">
                  <img
                    src="/assets/baseline_comparison.png"
                    alt="Baseline comparison"
                  />
                </div>
              </div>
            </section>
          </>
        )}

        {(view === "playground" || view === "judge") && (
          <>
            <section className="section mission">
              <SectionHead
                eyebrow="Mission"
                title="Live orchestrator state"
              />
              <div className="mission-grid">
                <StatBox label="Session" value={sessionId?.slice(0, 8) ?? "—"} />
                <StatBox label="Score" value={formatScore(info?.score)} />
                <StatBox
                  label="Budget"
                  value={
                    observation
                      ? `${observation.step_count}/${observation.max_steps}`
                      : "0/0"
                  }
                />
                <StatBox
                  label="Stakes"
                  value={
                    observation ? observation.stakes_level.toFixed(2) : "0.00"
                  }
                />
              </div>
              <div className="subtask-card">
                <div className="subtask-head">
                  <span>Current subtask</span>
                  <strong>{move.action}:{move.specialist}</strong>
                </div>
                <p>
                  {observation?.current_subtask ??
                    "Reset the episode to begin the live flow."}
                </p>
              </div>
            </section>

            <section className="section theater">
              <SectionHead
                eyebrow="Trust Surface"
                title="Public slots vs hidden risk"
              />
              <div className="theater-stage">
                {observation?.available_specialists.map((id, index) => {
                  const trust = observation.trust_snapshot[id] ?? 0.5;
                  return (
                    <motion.div
                      key={id}
                      className={`specialist-node ${
                        id === move.specialist ? "active" : ""
                      }`}
                      initial={{ opacity: 0, y: 14 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.35, delay: index * 0.05 }}
                    >
                      <div className="specialist-top">
                        <strong>{id}</strong>
                        <span>{trust.toFixed(2)}</span>
                      </div>
                      <div className="meter">
                        <span
                          style={{
                            width: `${trust * 100}%`,
                            background: trustColor(trust)
                          }}
                        />
                      </div>
                      <div className="specialist-foot">
                        {id === move.specialist ? "Recommended" : "Public slot"}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </section>

            <section className="section controls-panel">
              <SectionHead
                eyebrow="Command"
                title="Run the orchestrator"
              />
              <div className="action-stack">
                <button
                  className="btn primary wide"
                  disabled={running}
                  onClick={() => void autoRun("heuristic")}
                >
                  <Brain size={16} />
                  Run Heuristic
                </button>
                <button
                  className="btn wide"
                  disabled={running}
                  onClick={() => void autoRun("random")}
                >
                  <Cpu size={16} />
                  Run Random
                </button>
                <div className="micro-actions">
                  <button
                    className="btn micro"
                    disabled={running}
                    onClick={() => void stepEpisode("delegate", move.specialist)}
                  >
                    Delegate
                  </button>
                  <button
                    className="btn micro"
                    disabled={running}
                    onClick={() => void stepEpisode("verify", move.specialist)}
                  >
                    Verify
                  </button>
                  <button
                    className="btn micro"
                    disabled={running}
                    onClick={() => void stepEpisode("solve_independently")}
                  >
                    Self solve
                  </button>
                  <button
                    className="btn micro"
                    disabled={running}
                    onClick={() => void stepEpisode("skip")}
                  >
                    Skip
                  </button>
                </div>
              </div>
            </section>

            <section className="section payloads">
              <SectionHead
                eyebrow="Backend Visible"
                title="Request / response playground"
              />
              <div className="payload-grid">
                <JsonCard title="Last request" data={lastRequest} />
                <JsonCard title="Last response" data={lastResponse} />
              </div>
            </section>

            <section className="section recorder">
              <SectionHead
                eyebrow="Flight Recorder"
                title="Event trail"
              />
              <div className="event-list">
                {events.slice(-8).reverse().map((event) => (
                  <div className="event-row" key={`${event.step}-${event.action}`}>
                    <div className="event-step">#{event.step}</div>
                    <div className="event-copy">
                      <strong>{event.action}</strong>
                      <span>{event.summary}</span>
                    </div>
                    <div className="event-reward">{event.reward}</div>
                  </div>
                ))}
              </div>
            </section>
          </>
        )}

        {view === "judge" && (
          <section className="section judge">
            <SectionHead
              eyebrow="Judge Flow"
              title="A crisp 3-minute demo rail"
            />
            <div className="judge-sequence">
              <JudgeStep
                icon={ShieldAlert}
                title="1. Show the failure"
                copy="Run Random to show that long-horizon multi-agent routing collapses when trust is static."
              />
              <JudgeStep
                icon={Eye}
                title="2. Show the recovery"
                copy="Run Heuristic to show trust divergence, verification at risky gates, and stronger detection."
              />
              <JudgeStep
                icon={Sparkles}
                title="3. Show generalization"
                copy="Swap hidden profiles and replay to prove the orchestrator learned a transferable skill rather than a memorized slot."
              />
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

function SectionHead({
  eyebrow,
  title
}: {
  eyebrow: string;
  title: string;
}) {
  return (
    <div className="section-head">
      <span>{eyebrow}</span>
      <h3>{title}</h3>
    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value
}: {
  icon: typeof Activity;
  label: string;
  value: string;
}) {
  return (
    <div className="metric-card">
      <Icon size={16} />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function StoryLane({
  tone,
  label,
  title,
  steps
}: {
  tone: "before" | "after";
  label: string;
  title: string;
  steps: string[];
}) {
  return (
    <div className={`story-lane ${tone}`}>
      <div className="story-header">
        <span className="story-label">{label}</span>
        <strong>{title}</strong>
      </div>
      <div className="story-steps">
        {steps.map((step) => (
          <div key={step} className="story-step">
            {step}
          </div>
        ))}
      </div>
    </div>
  );
}

function ProofBar({
  label,
  value,
  tone
}: {
  label: string;
  value: number | undefined;
  tone: "bad" | "neutral" | "good" | "warm";
}) {
  const widths = typeof value === "number" ? `${Math.max(0, value * 100)}%` : "0%";
  return (
    <div className="proof-row">
      <span>{label}</span>
      <div className="proof-meter">
        <span className={`tone-${tone}`} style={{ width: widths }} />
      </div>
      <strong>{formatScore(value)}</strong>
    </div>
  );
}

function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-box">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function JsonCard({
  title,
  data
}: {
  title: string;
  data: Record<string, unknown> | null;
}) {
  return (
    <div className="json-card">
      <div className="json-head">{title}</div>
      <pre>{JSON.stringify(data ?? { waiting: true }, null, 2)}</pre>
    </div>
  );
}

function JudgeStep({
  icon: Icon,
  title,
  copy
}: {
  icon: typeof ArrowRight;
  title: string;
  copy: string;
}) {
  return (
    <motion.div
      className="judge-card"
      whileHover={{ y: -3 }}
      transition={{ duration: 0.18 }}
    >
      <Icon size={18} />
      <strong>{title}</strong>
      <span>{copy}</span>
    </motion.div>
  );
}

function HeroConstellation({ observation }: { observation: Observation | null }) {
  const ids = observation?.available_specialists ?? ["S0", "S1", "S2", "S3", "S4"];
  return (
    <div className="hero-constellation">
      <motion.div
        className="orb-core"
        animate={{ scale: [1, 1.06, 1], rotate: [0, 3, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
      />
      <div className="orb-ring ring-a" />
      <div className="orb-ring ring-b" />
      <div className="hero-node hero-node-main">
        <Brain size={18} />
        <span>Orchestrator</span>
      </div>
      {ids.map((id, index) => (
        <motion.div
          key={id}
          className={`hero-node hero-node-${index + 1}`}
          animate={{ y: [0, -6, 0] }}
          transition={{
            duration: 3 + index * 0.4,
            repeat: Infinity,
            ease: "easeInOut",
            delay: index * 0.2
          }}
        >
          <span>{id}</span>
        </motion.div>
      ))}
    </div>
  );
}
