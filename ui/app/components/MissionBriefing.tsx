"use client";
import type { Observation } from "../lib/types";
import { formatScore } from "../lib/theme";

export default function MissionBriefing({
  observation,
  score,
  detections,
  poisonings,
}: {
  observation: Observation | null;
  score: number | undefined;
  detections?: number;
  poisonings?: number;
}) {
  const stakes = observation?.stakes_level ?? 0;
  const highStakes = stakes >= 0.7;
  return (
    <>
      <div className="brief-grid">
        <div className="brief-stat">
          <div className="lbl">Score</div>
          <div className="val" style={{ color: "var(--green)" }}>{formatScore(score)}</div>
        </div>
        <div className="brief-stat">
          <div className="lbl">Budget</div>
          <div className="val">
            {observation ? `${observation.step_count}/${observation.max_steps}` : "—"}
          </div>
        </div>
        <div className="brief-stat">
          <div className="lbl">Detections</div>
          <div className="val" style={{ color: "var(--accent)" }}>{detections ?? 0}</div>
        </div>
        <div className="brief-stat">
          <div className="lbl">Poisonings</div>
          <div className="val" style={{ color: poisonings ? "var(--red)" : "var(--ink3)" }}>
            {poisonings ?? 0}
          </div>
        </div>
      </div>

      {/* stakes gauge */}
      <div className="stakes">
        <span style={{ fontSize: 11, color: "var(--ink3)", textTransform: "uppercase", letterSpacing: ".08em" }}>Stakes</span>
        <div className="stakes-track">
          <div
            className="stakes-fill"
            style={{
              width: `${stakes * 100}%`,
              background: highStakes
                ? "linear-gradient(90deg, var(--amber), var(--red))"
                : "linear-gradient(90deg, var(--accent), var(--green))",
            }}
          />
        </div>
        <span className="stakes-val">{stakes.toFixed(2)}</span>
        {highStakes && <span className="stakes-warn">⚠ HIGH</span>}
      </div>

      {/* current subtask */}
      <div className="brief-subtask">
        <div className="top">
          <span>Current Subtask</span>
          <span style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--ink)" }}>
            {observation ? `${observation.subtask_index + 1}/${observation.subtasks_total}` : "—"}
          </span>
        </div>
        <p>{observation?.current_subtask ?? "Reset the episode to begin."}</p>
      </div>
    </>
  );
}
