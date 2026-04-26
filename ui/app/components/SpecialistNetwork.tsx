"use client";
import { motion } from "framer-motion";
import { trustColor } from "../lib/theme";
import type { Observation } from "../lib/types";

const POS: [number, number][] = [
  [50, 10], [88, 35], [76, 80], [24, 80], [12, 35],
];
const CENTER: [number, number] = [50, 50];

export default function SpecialistNetwork({
  observation,
  trustDeltas,
  activeSpec,
}: {
  observation: Observation | null;
  trustDeltas: Record<string, number>;
  activeSpec: string | null;
}) {
  const ids = observation?.available_specialists || observation?.available_workers || ["S0", "S1", "S2", "S3", "S4"];
  return (
    <div className="net">
      <svg className="net-svg" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
        {ids.map((id, i) => {
          const [x, y] = POS[i];
          const isActive = id === activeSpec;
          return (
            <line
              key={id}
              className={`net-line${isActive ? " active" : ""}`}
              x1={CENTER[0]} y1={CENTER[1]}
              x2={x} y2={y}
              strokeDasharray={isActive ? "none" : "3 3"}
            />
          );
        })}
      </svg>

      {/* orchestrator */}
      <motion.div
        className="net-node orch"
        style={{ left: "50%", top: "50%" }}
        animate={{ scale: [1, 1.03, 1] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      >
        <div className="id">Orchestrator</div>
        <div className="trust" style={{ fontSize: 10, marginTop: 2 }}>
          {observation ? `Step ${observation.step_count}/${observation.max_steps}` : "—"}
        </div>
      </motion.div>

      {/* specialists */}
      {ids.map((id, i) => {
        const [x, y] = POS[i];
        const trust = observation?.trust_snapshot[id] ?? 0.5;
        const delta = trustDeltas[id] ?? 0;
        const isActive = id === activeSpec;
        const isDanger = trust < 0.3;
        return (
          <motion.div
            key={id}
            className={`net-node${isActive ? " active" : ""}${isDanger ? " danger" : ""}`}
            style={{ left: `${x}%`, top: `${y}%` }}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.06, duration: 0.3 }}
          >
            <div className="id">{id}</div>
            <div className="trust" style={{ color: trustColor(trust) }}>
              {trust.toFixed(2)}
            </div>
            {delta !== 0 && (
              <div className={`delta ${delta > 0 ? "delta-up" : "delta-down"}`}>
                {delta > 0 ? "▲" : "▼"} {Math.abs(delta).toFixed(2)}
              </div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}
