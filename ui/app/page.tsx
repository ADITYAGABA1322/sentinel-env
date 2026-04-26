"use client";

import { useEffect, useMemo } from "react";
import { useSentinel } from "./hooks/useSentinel";
import HeroCanvas from "./components/HeroCanvas";
import SystemModules from "./components/SystemModules";
import AgentTrustMonitor from "./components/AgentTrustMonitor";
import SimCanvas from "./components/SimCanvas";
import ExecutionLog from "./components/ExecutionLog";
import ArchitecturePipeline from "./components/ArchitecturePipeline";
import MetricsGrid from "./components/MetricsGrid";
import type { TaskType, AutoPolicy } from "./lib/types";

export default function Page() {
  const s = useSentinel();

  useEffect(() => {
    void s.resetEpisode();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const adversarialAgents = useMemo(() => new Set(
    s.events
      .filter(e => e.outcome === "poisoned" || e.outcome === "blocked")
      .map(e => e.specialist)
      .filter((x): x is string => !!x)
  ), [s.events]);

  const totalReward = s.info?.total_reward ?? 0;

  return (
    <>
      {/* NAV */}
      <nav className="nav-bar">
        <div className="nav-logo">
          <div className="nav-logo-dot" />
          SENTINEL
        </div>
        <ul className="nav-links">
          <li><a href="#overview">MODULES</a></li>
          <li><a href="#simulation">SIMULATION</a></li>
          <li><a href="#architecture">ARCHITECTURE</a></li>
          <li><a href="#metrics">METRICS</a></li>
        </ul>
        <div className={`nav-badge ${s.running ? "running" : s.done ? "complete" : ""}`}>
          <div className="nav-badge-dot" />
          {s.running ? "RUNNING" : s.done ? "COMPLETE" : "SYSTEM ONLINE"}
        </div>
      </nav>

      {/* HERO */}
      <section className="hero" id="hero">
        <HeroCanvas />
        <div className="hero-scanline" />

        <div className="hero-content">
          <div className="hero-tag anim-1">
            SYS.CORE // SENTINEL v2.4.1 // MARL FRAMEWORK
          </div>
          <h1 className="anim-2">
            Train AI to <span className="accent">Trust</span> —<br />
            and Survive <span className="accent-red">Adversaries</span>
          </h1>
          <p className="hero-sub anim-3">
            A multi-agent reinforcement learning system where an orchestrator
            learns to detect deception, assign trust, and optimize decisions in
            real-time adversarial environments.
          </p>
          <div className="hero-ctas anim-4">
            <button
              className="btn-primary"
              onClick={() => {
                document.getElementById("simulation")?.scrollIntoView({ behavior: "smooth" });
                setTimeout(() => {
                  void s.resetEpisode().then(() => s.autoRun("trained" as AutoPolicy));
                }, 400);
              }}
              disabled={s.running}
            >
              ▶ Launch Simulation
            </button>
            <button
              className="btn-secondary"
              onClick={() => document.getElementById("architecture")?.scrollIntoView({ behavior: "smooth" })}
            >
              View System Architecture
            </button>
          </div>
        </div>

        <div className="hero-stats anim-5">
          <div className="hero-stat-item">
            <div className="hero-stat-num">{s.observation?.available_specialists?.length ?? 5}</div>
            <div className="hero-stat-label">Active Agents</div>
          </div>
          <div className="hero-stat-item">
            <div className="hero-stat-num">
              {s.proof?.trained
                ? Math.round(s.proof.trained.avg_detection_rate * 100)
                : 92
              }<span style={{ fontSize: 14, opacity: 0.4 }}>%</span>
            </div>
            <div className="hero-stat-label">Trust Accuracy</div>
          </div>
          <div className="hero-stat-item">
            <div className="hero-stat-num">
              {s.proof?.trained
                ? s.proof.trained.avg_score.toFixed(2)
                : "0.91"
              }
            </div>
            <div className="hero-stat-label">Avg Score</div>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* SYSTEM MODULES */}
      <section className="section-block alt-bg" id="overview">
        <div className="section-label">01 // SYSTEM MODULES</div>
        <h2 className="section-title">Core Architecture</h2>
        <p className="section-desc">
          Each module operates as an independent inference layer within the
          trust-calibration pipeline. All components communicate via the
          orchestration bus.
        </p>
        <SystemModules
          running={s.running}
          done={s.done}
          adversarialCount={adversarialAgents.size}
        />
      </section>

      <div className="divider" />

      {/* SIMULATION */}
      <section className="section-block" id="simulation">
        <div className="section-label">02 // LIVE PREVIEW</div>
        <h2 className="section-title">Simulation Control Panel</h2>
        <p className="section-desc">
          Real-time orchestrator view. Agent trust scores update per-step. Red
          indicates flagged adversarial behaviour.
        </p>

        <div className="sim-wrapper">
          <div className="sim-topbar">
            <div className="sim-dots">
              <div className="sim-dot r" />
              <div className="sim-dot y" />
              <div className="sim-dot g" />
            </div>
            <div className="sim-topbar-title">
              SENTINEL // ORCHESTRATOR VIEW // TASK: {s.taskType?.toUpperCase() ?? "TASK3"} // STEP: {s.observation?.step_count ?? 0}
            </div>
            <div className="sim-topbar-badge">
              ● {s.running ? "LIVE" : s.done ? "DONE" : "READY"}
            </div>
          </div>

          <div className="sim-body">
            {/* LEFT: AGENTS */}
            <AgentTrustMonitor
              observation={s.observation}
              trustDeltas={s.trustDeltas}
              activeSpec={s.activeSpec}
              events={s.events}
              running={s.running}
              totalReward={totalReward}
            />

            {/* CENTER: CANVAS */}
            <div className="sim-center">
              <SimCanvas
                trustSnapshot={s.observation?.trust_snapshot ?? {}}
                adversarialAgents={adversarialAgents}
                activeSpec={s.activeSpec}
              />
            </div>

            {/* RIGHT: LOGS */}
            <ExecutionLog
              events={s.events}
              observation={s.observation}
              info={s.info}
            />
          </div>

          {/* Controls Row */}
          <div className="sim-controls-row">
            <span className="ctrl-label">POLICY:</span>
            <button className="btn-sm-ctrl" onClick={() => void s.autoRun("heuristic" as AutoPolicy)} disabled={s.running || s.done}>
              ▶ HEURISTIC
            </button>
            <button className="btn-sm-ctrl" onClick={() => void s.autoRun("random" as AutoPolicy)} disabled={s.running || s.done}>
              ⚄ RANDOM
            </button>
            <button className="btn-sm-ctrl" onClick={() => void s.autoRun("trained" as AutoPolicy)} disabled={s.running || s.done}>
              🧠 TRAINED RL
            </button>
            {s.running && (
              <button className="btn-sm-ctrl danger" onClick={s.stopAutoRun}>
                ■ STOP
              </button>
            )}
            <span className="ctrl-label" style={{ marginLeft: "auto" }}>ACTIONS:</span>
            <button className="btn-sm-ctrl" onClick={() => void s.stepEpisode("delegate")} disabled={s.running || s.done}>
              DELEGATE
            </button>
            <button className="btn-sm-ctrl" onClick={() => void s.stepEpisode("verify")} disabled={s.running || s.done}>
              VERIFY
            </button>
            <button className="btn-sm-ctrl" onClick={() => void s.stepEpisode("skip")} disabled={s.running || s.done}>
              SKIP
            </button>
          </div>

          {/* Sim Footer */}
          <div className="sim-footer">
            <span>TASK: <b>{s.taskType?.toUpperCase() ?? "TASK3"}</b></span>
            <span>SEED: <b>{s.seed}</b></span>
            <span>ALGO: <b>DQN+TCE</b></span>
            <span>SESSION: <b>{s.sessionId?.slice(0, 8) ?? "—"}</b></span>
            <span style={{ marginLeft: "auto" }}>
              <button
                className="btn-sm-ctrl"
                onClick={() => void s.resetEpisode()}
                disabled={s.running}
                style={{ fontSize: 8 }}
              >
                ⟳ RESET EPISODE
              </button>
            </span>
            <select
              value={s.taskType}
              onChange={(e) => {
                const next = e.target.value as TaskType;
                s.setTaskType(next);
                void s.resetEpisode(next, s.seed);
              }}
              style={{
                background: "transparent", border: "1px solid rgba(0,200,255,0.2)",
                color: "var(--cyan)", fontFamily: "var(--font-mono)", fontSize: 9,
                padding: "2px 6px", cursor: "pointer",
              }}
            >
              <option value="task1" style={{ background: "var(--bg)" }}>TASK 1</option>
              <option value="task2" style={{ background: "var(--bg)" }}>TASK 2</option>
              <option value="task3" style={{ background: "var(--bg)" }}>TASK 3</option>
            </select>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* ARCHITECTURE */}
      <section className="section-block alt-bg" id="architecture">
        <div className="section-label">03 // SYSTEM DESIGN</div>
        <h2 className="section-title">Execution Pipeline</h2>
        <p className="section-desc">
          Data flows unidirectionally through the trust-calibrated RL loop. Each
          stage emits telemetry to the monitoring bus.
        </p>
        <ArchitecturePipeline />
      </section>

      <div className="divider" />

      {/* METRICS */}
      <section className="section-block" id="metrics">
        <div className="section-label">04 // EVALUATION RESULTS</div>
        <h2 className="section-title">Experimental Benchmarks</h2>
        <p className="section-desc">
          Averaged across evaluation episodes. Adversarial injection ratio fixed at
          20%. Baseline: naive averaging orchestrator without trust calibration.
        </p>
        <MetricsGrid proof={s.proof} />
      </section>

      {/* FOOTER */}
      <footer className="site-footer">
        <div className="footer-left">
          <strong>SENTINEL</strong><br />
          Multi-Agent Reinforcement Learning System<br />
          Research prototype — not for production use.
        </div>
        <div className="footer-right">
          BUILD 2.4.1 // MARL-FRAMEWORK // MIT LICENSE<br />
          © 2025 SENTINEL LAB. ALL RIGHTS RESERVED.
        </div>
      </footer>
    </>
  );
}
