"use client";

import { useRef, useEffect } from "react";
import type { EventItem, Observation } from "../lib/types";

type Props = {
  events: EventItem[];
  observation: Observation | null;
  info: { total_reward: number; score: number; adversarial_detections?: number; adversarial_poisonings?: number } | undefined;
};

function logClass(e: EventItem): string {
  if (e.outcome === "poisoned") return "alert";
  if (e.outcome === "blocked") return "warn";
  if (e.outcome === "success") return "ok";
  return "";
}

function formatTime(step: number): string {
  const m = 14, s = 3 + step;
  const ss = s % 60;
  const mm = m + Math.floor(s / 60);
  return `${mm}:${String(ss).padStart(2, "0")}:${String(Math.floor(Math.random() * 60)).padStart(2, "0")}`;
}

export default function ExecutionLog({ events, observation, info }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [events.length]);

  const detectRate = info?.adversarial_detections !== undefined && info?.adversarial_poisonings !== undefined
    ? (info.adversarial_detections + info.adversarial_poisonings) > 0
      ? Math.round((info.adversarial_detections / (info.adversarial_detections + info.adversarial_poisonings)) * 100)
      : 0
    : null;

  return (
    <div className="sim-panel" style={{ borderRight: "none", borderLeft: "1px solid rgba(0,200,255,0.08)" }}>
      <div className="sim-panel-label">EVENT LOG</div>

      <div ref={scrollRef} style={{ maxHeight: 220, overflowY: "auto" }}>
        {events.slice(-12).map((e, i) => (
          <div key={i} className={`log-entry ${logClass(e)}`}>
            <span className="log-time">{formatTime(e.step)}</span>
            {e.action === "reset"
              ? "EPISODE RESET ⟳"
              : `${e.specialist ?? "SYS"} ${e.summary.substring(0, 40)}`
            }
          </div>
        ))}
        {events.length === 0 && (
          <div className="log-entry" style={{ opacity: 0.3 }}>
            Waiting for simulation data...
          </div>
        )}
      </div>

      <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid rgba(0,200,255,0.08)" }}>
        <div className="sim-panel-label">EPISODE METRICS</div>
        <div className="sim-metric-row">
          <span className="sim-metric-label">CUMULATIVE REWARD</span>
          <span className={`sim-metric-val ${(info?.total_reward ?? 0) >= 0 ? "g" : "r"}`}>
            {(info?.total_reward ?? 0) >= 0 ? "+" : ""}{(info?.total_reward ?? 0).toFixed(2)}
          </span>
        </div>
        <div className="sim-metric-row">
          <span className="sim-metric-label">SCORE</span>
          <span className="sim-metric-val c">{(info?.score ?? 0).toFixed(3)}</span>
        </div>
        {detectRate !== null && (
          <div className="sim-metric-row">
            <span className="sim-metric-label">DETECT RATE</span>
            <span className="sim-metric-val c">{detectRate}%</span>
          </div>
        )}
        <div className="sim-metric-row">
          <span className="sim-metric-label">STEP</span>
          <span className="sim-metric-val a">
            {observation?.step_count ?? 0}/{observation?.max_steps ?? 0}
          </span>
        </div>
      </div>
    </div>
  );
}
