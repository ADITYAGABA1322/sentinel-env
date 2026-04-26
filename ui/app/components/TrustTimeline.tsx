"use client";
import { motion } from "framer-motion";
import { trustColor } from "../lib/theme";
import type { Observation } from "../lib/types";

export default function TrustTimeline({
  observation,
  trustDeltas,
}: {
  observation: Observation | null;
  trustDeltas: Record<string, number>;
}) {
<<<<<<< HEAD
  const ids = observation?.available_specialists ?? ["S0", "S1", "S2", "S3", "S4"];
=======
  const ids = observation?.available_specialists ?? observation?.available_workers ?? ["S0", "S1", "S2", "S3", "S4"];
>>>>>>> a89a58750afb4cf3e8d49f13fe66d7c227911387
  return (
    <div className="tl">
      {ids.map((id) => {
        const trust = observation?.trust_snapshot[id] ?? 0.5;
        const delta = trustDeltas[id] ?? 0;
        return (
          <div className="tl-row" key={id}>
            <span className="tl-id">{id}</span>
            <div className="tl-track">
              <motion.div
                className="tl-fill"
                style={{ background: trustColor(trust) }}
                animate={{ width: `${Math.max(2, trust * 100)}%` }}
                transition={{ type: "spring", stiffness: 260, damping: 24 }}
              />
            </div>
            <span className="tl-val" style={{ color: trustColor(trust) }}>
              {trust.toFixed(2)}
            </span>
            <span className={`tl-delta ${delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : ""}`}>
              {delta !== 0 ? `${delta > 0 ? "+" : ""}${delta.toFixed(2)}` : ""}
            </span>
          </div>
        );
      })}
    </div>
  );
}
