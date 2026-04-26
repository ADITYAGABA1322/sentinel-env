"use client";
import { Shield, Eye, Wrench, SkipForward, Brain, Cpu, Square, Sparkles } from "lucide-react";
import type { ActionType, AutoPolicy } from "../lib/types";

const ACTIONS: { id: ActionType; label: string; icon: typeof Shield }[] = [
  { id: "delegate",            label: "Delegate",   icon: Shield },
  { id: "verify",              label: "Verify",     icon: Eye },
  { id: "solve_independently", label: "Self Solve", icon: Wrench },
  { id: "skip",                label: "Skip",       icon: SkipForward },
];

export default function ActionCenter({
  recommended,
  running,
  done,
  onStep,
  onAutoRun,
  onStop,
}: {
  recommended: { action: ActionType; specialist: string; trust: number };
  running: boolean;
  done: boolean;
  onStep: (action: ActionType) => void;
  onAutoRun: (policy: AutoPolicy) => void;
  onStop: () => void;
}) {
  return (
    <>
      <div className="ac-grid">
        {ACTIONS.map((a) => {
          const isRec = a.id === recommended.action;
          return (
            <button
              key={a.id}
              className={`ac-btn${isRec ? " rec" : ""}`}
              disabled={running || done}
              onClick={() => onStep(a.id)}
            >
              <a.icon size={16} />
              {a.label}
              {isRec && a.id !== "skip" && (
                <span style={{ fontSize: 10, color: "var(--ink3)" }}>→ {recommended.specialist}</span>
              )}
            </button>
          );
        })}
      </div>
      <div className="ac-auto">
        {running ? (
          <button className="btn btn-danger btn-block" onClick={onStop}>
            <Square size={14} /> Stop
          </button>
        ) : (
          <>
            <button className="btn btn-primary btn-block" disabled={running} onClick={() => onAutoRun("heuristic")}>
              <Brain size={14} /> Auto Heuristic
            </button>
            <button className="btn btn-primary btn-block" disabled={running} onClick={() => onAutoRun("trained")}>
              <Sparkles size={14} /> Auto GRPO Replay
            </button>
            <button className="btn btn-block" disabled={running} onClick={() => onAutoRun("random")}>
              <Cpu size={14} /> Auto Random
            </button>
          </>
        )}
      </div>
    </>
  );
}
