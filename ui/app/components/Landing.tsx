"use client";
import { Brain, Shuffle, CircleGauge, ShieldAlert, ArrowRight, Sparkles } from "lucide-react";
import { formatScore } from "../lib/theme";
import type { EvalSummary } from "../lib/types";

const ARCH = [
  { icon: Brain,               title: "Orchestrator",        desc: "Learns trust, verification, and recovery from behavior alone." },
  { icon: Shuffle,             title: "Shuffled Specialists", desc: "Hidden profiles reshuffle every reset — no identity memorization." },
  { icon: CircleGauge,         title: "Trust Ledger",        desc: "Bayesian updates turn observed behavior into routing signal." },
  { icon: ShieldAlert,         title: "Reward Engine",       desc: "Completion + detection + calibration + efficiency." },
];

const BEFORE_STEPS = [
  "Orchestrator delegates with no evidence.",
  "Adversarial specialist poisons high-stakes output.",
  "Poisoned state cascades across downstream tasks.",
  "Mission fails — nobody knows which slot was risky.",
];
const AFTER_STEPS = [
  "Trust ledger updates after every action.",
  "High-stakes + low-trust triggers verification.",
  "Adversarial attempt blocked before cascade.",
  "Profile swap proves skill, not memorized identity.",
];

export default function Landing({
  proof,
  onEnterMission,
  onEnterJudge,
}: {
  proof: {
    random: EvalSummary;
    heuristic: EvalSummary;
    oracle: EvalSummary;
    trained?: EvalSummary;
    task3Heuristic: EvalSummary;
  } | null;
  onEnterMission: () => void;
  onEnterJudge: () => void;
}) {
  return (
    <div className="land">
      {/* hero */}
      <div className="land-hero">
        <h1>
          Agents fail because they{" "}
          <span>trust blindly</span>
        </h1>
        <p>
          SENTINEL trains an orchestrator to decide who to trust, when to verify,
          and how to recover in long multi-agent tasks when specialist agents are
          unreliable or adversarial.
        </p>
        <div className="land-ctas">
          <button className="btn btn-primary btn-lg" onClick={onEnterMission}>
            <Sparkles size={16} /> Try It Live
          </button>
          <button className="btn btn-lg" onClick={onEnterJudge}>
            <ArrowRight size={16} /> Judge Demo
          </button>
        </div>
      </div>

      {/* score strip */}
      <div className="score-strip">
        <div className="panel score-card r">
          <div className="lbl">Random</div>
          <div className="val">{formatScore(proof?.random.avg_score)}</div>
        </div>
        <div className="panel score-card a">
          <div className="lbl">Heuristic</div>
          <div className="val">{formatScore(proof?.heuristic.avg_score)}</div>
        </div>
        <div className="panel score-card g">
          <div className="lbl">Oracle‑lite</div>
          <div className="val">{formatScore(proof?.oracle.avg_score)}</div>
        </div>
        <div className="panel score-card a">
          <div className="lbl">GRPO Replay</div>
          <div className="val">{formatScore(proof?.trained?.avg_score)}</div>
        </div>
        <div className="panel score-card w">
          <div className="lbl">Task 3 Detect</div>
          <div className="val">{formatScore(proof?.task3Heuristic.avg_detection_rate)}</div>
        </div>
      </div>

      {/* before / after */}
      <div className="ba-section">
        <div className="panel-head" style={{ textAlign: "center", marginBottom: 20 }}>
          <div className="panel-eyebrow">Why This Matters</div>
          <div className="panel-title">Before vs After SENTINEL</div>
        </div>
        <div className="ba-grid">
          <div className="panel ba-card before">
            <div className="ba-tag">✗ Without Trust Calibration</div>
            <h3>Blind Delegation</h3>
            <div className="ba-steps">
              {BEFORE_STEPS.map((s, i) => (
                <div className="ba-step" key={i}>
                  <span className="num">{i + 1}</span>
                  <span>{s}</span>
                </div>
              ))}
            </div>
            <div className="ba-score">0.19</div>
          </div>
          <div className="panel ba-card after">
            <div className="ba-tag">✓ With SENTINEL Training</div>
            <h3>Trust‑Aware Routing</h3>
            <div className="ba-steps">
              {AFTER_STEPS.map((s, i) => (
                <div className="ba-step" key={i}>
                  <span className="num">{i + 1}</span>
                  <span>{s}</span>
                </div>
              ))}
            </div>
            <div className="ba-score">0.71</div>
          </div>
        </div>
      </div>

      {/* architecture */}
      <div style={{ marginTop: 8 }}>
        <div className="panel-head" style={{ textAlign: "center", marginBottom: 16 }}>
          <div className="panel-eyebrow">Architecture</div>
          <div className="panel-title">What the System Is Made Of</div>
        </div>
        <div className="arch-grid">
          {ARCH.map((a) => (
            <div className="panel arch-card" key={a.title}>
              <a.icon size={20} />
              <h4>{a.title}</h4>
              <p>{a.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
