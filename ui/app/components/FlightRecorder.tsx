"use client";
import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { EventItem } from "../lib/types";

const ICONS: Record<EventItem["outcome"], string> = {
  success:  "✅",
  blocked:  "🛡️",
  poisoned: "☠️",
  skipped:  "⏭️",
  reset:    "🔄",
};

export default function FlightRecorder({
  events,
  lastReq,
  lastRes,
}: {
  events: EventItem[];
  lastReq: Record<string, unknown> | null;
  lastRes: Record<string, unknown> | null;
}) {
  const [showJson, setShowJson] = useState(false);
  const recent = events.slice(-10).reverse();

  return (
    <>
      <div className="fr-list">
        <AnimatePresence initial={false}>
          {recent.map((ev) => (
            <motion.div
              key={`${ev.step}-${ev.action}`}
              className="fr-row"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2 }}
            >
              <span className="fr-step">#{ev.step}</span>
              <div>
                <span className="fr-action">
                  <span className="fr-icon">{ICONS[ev.outcome]} </span>
                  {ev.action}{ev.specialist ? `:${ev.specialist}` : ""}
                </span>
                <div className="fr-summary">{ev.summary}</div>
              </div>
              <span className={`fr-reward ${ev.reward >= 0.5 ? "pos" : "neg"}`}>
                {ev.outcome === "reset" ? "—" : ev.reward.toFixed(2)}
              </span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      <button className="fr-toggle" onClick={() => setShowJson(!showJson)}>
        {showJson ? "Hide" : "Show"} raw JSON
      </button>

      {showJson && (
        <div className="json-view">
          <pre>{JSON.stringify({ request: lastReq, response: lastRes }, null, 2)}</pre>
        </div>
      )}
    </>
  );
}
