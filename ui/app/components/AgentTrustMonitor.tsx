"use client";

import type { Observation, EventItem } from "../lib/types";

const AGENT_ROLES: Record<string, string> = {
  S0: "COORDINATOR", S1: "OBSERVER", S2: "EXECUTOR", S3: "FLAGGED", S4: "VALIDATOR",
};

type Props = {
  observation: Observation | null;
  trustDeltas: Record<string, number>;
  activeSpec: string | null;
  events: EventItem[];
  running: boolean;
  totalReward: number;
};

function trustClass(t: number) {
  if (t >= 0.65) return "high";
  if (t >= 0.35) return "mid";
  return "low";
}

function trustColor(t: number) {
  if (t >= 0.65) return "var(--green)";
  if (t >= 0.35) return "var(--amber)";
  return "var(--red)";
}

export default function AgentTrustMonitor({
  observation, trustDeltas, activeSpec, events, running, totalReward,
}: Props) {
  const agents = observation?.available_specialists || observation?.available_workers || ["S0", "S1", "S2", "S3", "S4"];
  const trust = observation?.trust_snapshot ?? {};
  const lastReward = observation?.last_reward ?? 0;

  // detect adversarial from events
  const poisonedAgents = new Set(
    events.filter(e => e.outcome === "poisoned" || e.outcome === "blocked")
      .map(e => e.specialist).filter(Boolean)
  );

  // mean trust
  const vals = agents.map(id => trust[id] ?? 0.5);
  const meanTrust = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0.5;
  const advRatio = agents.length ? Math.round((poisonedAgents.size / agents.length) * 100) : 0;

  return (
    <div className="sim-panel">
      <div className="sim-panel-label">AGENT TRUST REGISTRY</div>

      {agents.map(id => {
        const t = trust[id] ?? 0.5;
        const delta = trustDeltas[id] ?? 0;
        const isActive = activeSpec === id;
        const isAdv = poisonedAgents.has(id);
        const role = AGENT_ROLES[id] ?? "AGENT";

        return (
          <div className="agent-row" key={id}>
            <div className="agent-header">
              <span className={`agent-id ${isActive ? "active" : isAdv ? "adversarial" : "neutral"}`}>
                {id} // {isAdv ? "⚠ FLAGGED" : role}
              </span>
              <span className="agent-trust-val" style={{ color: trustColor(t) }}>
                {t.toFixed(2)}
              </span>
            </div>
            <div className="trust-bar-bg">
              <div
                className={`trust-bar-fill ${trustClass(t)}`}
                style={{ width: `${Math.max(2, t * 100)}%` }}
              />
            </div>
            <div className="agent-state" style={isAdv ? { color: "var(--red)" } : {}}>
              {isAdv
                ? "STATE: ADVERSARIAL // ISOLATED"
                : isActive
                  ? "STATE: ACTIVE // DELEGATING"
                  : `STATE: READY // Δ ${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`
              }
            </div>
          </div>
        );
      })}

      <div style={{ marginTop: 20, paddingTop: 16, borderTop: "1px solid rgba(0,200,255,0.08)" }}>
        <div className="sim-metric-row">
          <span className="sim-metric-label">MEAN TRUST</span>
          <span className="sim-metric-val c">{meanTrust.toFixed(3)}</span>
        </div>
        <div className="sim-metric-row">
          <span className="sim-metric-label">ADV RATIO</span>
          <span className="sim-metric-val r">{advRatio}%</span>
        </div>
        <div className="sim-metric-row">
          <span className="sim-metric-label">STEP REWARD</span>
          <span className={`sim-metric-val ${lastReward >= 0 ? "g" : "r"}`}>
            {lastReward >= 0 ? "+" : ""}{lastReward.toFixed(3)}
          </span>
        </div>
        <div className="sim-metric-row">
          <span className="sim-metric-label">TOTAL REWARD</span>
          <span className={`sim-metric-val ${totalReward >= 0 ? "g" : "r"}`}>
            {totalReward >= 0 ? "+" : ""}{totalReward.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}
