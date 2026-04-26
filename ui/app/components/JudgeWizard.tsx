"use client";
import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldAlert, Eye, Sparkles, RotateCcw, Play, Square } from "lucide-react";
import { trustColor } from "../lib/theme";
import type { AutoPolicy, StepResult, TaskType, Observation, EventItem } from "../lib/types";

type JudgePhase = 0 | 1 | 2 | 3;

interface PhaseResult {
  score: number;
  detections: number;
  poisonings: number;
  steps: number;
  finalTrust: Record<string, number>;
  events: EventItem[];
}

const STEPS = [
  {
    icon: ShieldAlert,
    num: "Step 1 of 3",
    title: "Show the Failure",
    desc: "The orchestrator delegates blindly using a random policy. No trust model. No verification. Watch as adversarial agents poison the mission unchecked.",
    btnLabel: "Run Random Policy",
    color: "var(--red)",
  },
  {
    icon: Eye,
    num: "Step 2 of 3",
    title: "Show the Recovery",
    desc: "Now the orchestrator uses behavioral trust. It routes to trusted specialists, triggers verification when stakes are high, and catches adversarial attempts before they cascade.",
    btnLabel: "Run Heuristic Policy",
    color: "var(--green)",
  },
  {
    icon: Sparkles,
    num: "Step 3 of 3",
    title: "Prove Generalization",
    desc: "Hidden profiles are reshuffled. The adversarial agent moves to a different slot. The orchestrator re-learns trust from scratch — proving this is a skill, not memorized identity.",
    btnLabel: "Swap Profiles & Replay",
    color: "var(--accent)",
  },
];

export default function JudgeWizard({
  autoRun,
  resetEpisode,
  swapProfiles,
  observation,
  events,
  info,
  running: globalRunning,
}: {
  autoRun: (policy: AutoPolicy) => Promise<void>;
  resetEpisode: (task?: TaskType, seed?: number) => Promise<StepResult | null>;
  swapProfiles: () => Promise<void>;
  observation: Observation | null;
  events: EventItem[];
  info: StepResult["info"] | undefined;
  running: boolean;
}) {
  const [phase, setPhase] = useState<JudgePhase>(0);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<(PhaseResult | null)[]>([null, null, null]);
  const eventsRef = useRef(events);
  const infoRef = useRef(info);
  const obsRef = useRef(observation);

  useEffect(() => { eventsRef.current = events; }, [events]);
  useEffect(() => { infoRef.current = info; }, [info]);
  useEffect(() => { obsRef.current = observation; }, [observation]);

  const captureResult = useCallback((): PhaseResult => ({
    score: infoRef.current?.score ?? 0,
    detections: infoRef.current?.adversarial_detections ?? 0,
    poisonings: infoRef.current?.adversarial_poisonings ?? 0,
    steps: infoRef.current?.step_count ?? 0,
    finalTrust: obsRef.current?.trust_snapshot ?? {},
    events: [...eventsRef.current],
  }), []);

  const runPhase = useCallback(async () => {
    setRunning(true);
    try {
      if (phase === 0) {
        await resetEpisode();
        await autoRun("random");
        const r = captureResult();
        setResults((p) => { const n = [...p]; n[0] = r; return n; });
        setPhase(1);
      } else if (phase === 1) {
        await resetEpisode();
        await autoRun("heuristic");
        const r = captureResult();
        setResults((p) => { const n = [...p]; n[1] = r; return n; });
        setPhase(2);
      } else if (phase === 2) {
        await swapProfiles();
        await autoRun("trained");
        const r = captureResult();
        setResults((p) => { const n = [...p]; n[2] = r; return n; });
        setPhase(3);
      }
    } finally {
      setRunning(false);
    }
  }, [phase, autoRun, resetEpisode, swapProfiles, captureResult]);

  const restart = () => {
    setPhase(0);
    setResults([null, null, null]);
  };

  const currentStep = Math.min(phase, 2);
  const step = STEPS[currentStep];
  const Icon = step.icon;
  const isRunning = running || globalRunning;

  // Live trust data during run
  const trustEntries = observation
    ? Object.entries(observation.trust_snapshot).sort(([a], [b]) => a.localeCompare(b))
    : [];

  return (
    <div className="jw">
      {/* progress dots */}
      <div className="jw-progress">
        {[0, 1, 2].map((i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div className={`jw-dot ${phase > i ? "done" : phase === i ? "active" : ""}`} />
            {i < 2 && <div className={`jw-bar ${phase > i ? "done" : ""}`} />}
          </div>
        ))}
      </div>

      {/* main stage */}
      <AnimatePresence mode="wait">
        <motion.div
          key={phase}
          className="panel jw-stage"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
        >
          {phase < 3 ? (
            <>
              <div className="jw-step-num">{step.num}</div>
              <h2>
                <Icon size={28} style={{ verticalAlign: "middle", marginRight: 10, color: step.color }} />
                {step.title}
              </h2>
              <p>{step.desc}</p>

              <button
                className="btn btn-primary btn-lg btn-glow"
                disabled={isRunning}
                onClick={runPhase}
              >
                {isRunning ? (
                  <><Square size={16} /> Running…</>
                ) : (
                  <><Play size={16} /> {step.btnLabel}</>
                )}
              </button>

              {/* Show PREVIOUS result if we have one (comparison view) */}
              {phase === 1 && results[0] && (
                <div style={{ marginTop: 28 }}>
                  <div style={{ fontSize: 11, color: "var(--ink3)", textTransform: "uppercase", letterSpacing: ".1em", fontWeight: 700, marginBottom: 12, fontFamily: "var(--mono)" }}>
                    Previous: Random Policy Result
                  </div>
                  <PhaseResultCard result={results[0]} variant="bad" />
                </div>
              )}

              {phase === 2 && results[0] && results[1] && (
                <div style={{ marginTop: 28 }}>
                  <div style={{ fontSize: 11, color: "var(--ink3)", textTransform: "uppercase", letterSpacing: ".1em", fontWeight: 700, marginBottom: 12, fontFamily: "var(--mono)" }}>
                    Before vs After Comparison
                  </div>
                  <div className="jw-compare">
                    <div className="panel jw-compare-card bad">
                      <h4>Random (Blind)</h4>
                      <div className="big">{results[0].score.toFixed(3)}</div>
                      <div style={{ marginTop: 8, fontSize: 12, color: "var(--ink3)" }}>
                        {results[0].poisonings} poisonings · {results[0].detections} detections
                      </div>
                    </div>
                    <div className="panel jw-compare-card good">
                      <h4>Heuristic (Trust)</h4>
                      <div className="big">{results[1].score.toFixed(3)}</div>
                      <div style={{ marginTop: 8, fontSize: 12, color: "var(--ink3)" }}>
                        {results[1].poisonings} poisonings · {results[1].detections} detections
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            /* completion screen */
            <>
              <div className="jw-step-num">Demo Complete</div>
              <h2>
                <Sparkles size={28} style={{ verticalAlign: "middle", marginRight: 10, color: "var(--green)" }} />
                Trust Calibration Works
              </h2>
              <p>
                Across all three runs, the orchestrator learned to identify and route around adversarial agents — even when specialist identities were reshuffled.
              </p>

              {/* three-way comparison */}
              <div className="jw-results" style={{ maxWidth: 600, gridTemplateColumns: "repeat(3,1fr)" }}>
                {["Random", "Heuristic", "After Swap"].map((label, i) => {
                  const r = results[i];
                  return (
                    <div className="jw-stat" key={label}>
                      <div className="lbl">{label}</div>
                      <div className="val" style={{
                        color: i === 0 ? "var(--red)" : "var(--green)",
                        textShadow: i === 0 ? "0 0 20px var(--glow-red)" : "0 0 20px var(--glow-green)",
                      }}>
                        {r ? r.score.toFixed(3) : "—"}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Final trust comparison */}
              {results[1] && results[2] && (
                <div className="jw-inline">
                  <div className="panel" style={{ textAlign: "left" }}>
                    <div className="panel-eyebrow">Heuristic Final Trust</div>
                    <TrustBars trust={results[1].finalTrust} />
                  </div>
                  <div className="panel" style={{ textAlign: "left" }}>
                    <div className="panel-eyebrow">After Swap Final Trust</div>
                    <TrustBars trust={results[2].finalTrust} />
                  </div>
                </div>
              )}

              <div className="jw-nav">
                <button className="btn btn-lg btn-glow" onClick={restart}>
                  <RotateCcw size={16} /> Run Again
                </button>
              </div>
            </>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Live data panel - shows during runs */}
      {isRunning && observation && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="jw-inline"
        >
          <div className="panel" style={{ textAlign: "left" }}>
            <div className="panel-eyebrow">Live Trust Scores</div>
            <TrustBars trust={observation.trust_snapshot} />
          </div>
          <div className="panel" style={{ textAlign: "left" }}>
            <div className="panel-eyebrow">Live Events</div>
            <div style={{ display: "grid", gap: 4, maxHeight: 200, overflow: "auto", fontSize: 12 }}>
              {events.slice(-8).reverse().map((ev, i) => (
                <div key={i} style={{
                  padding: "6px 8px", borderRadius: 6,
                  background: "rgba(0,0,0,.2)", color: "var(--ink2)",
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                }}>
                  <span>
                    <span style={{ color: "var(--ink3)", fontFamily: "var(--mono)" }}>#{ev.step}</span>{" "}
                    {ev.action}{ev.specialist ? `:${ev.specialist}` : ""}
                  </span>
                  <span style={{
                    fontFamily: "var(--mono)", fontWeight: 700,
                    color: ev.reward >= 0.5 ? "var(--green)" : "var(--red)",
                  }}>
                    {ev.outcome === "reset" ? "—" : ev.reward.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

/* ── helper components ─────────────────────────────── */

function TrustBars({ trust }: { trust: Record<string, number> }) {
  const entries = Object.entries(trust).sort(([a], [b]) => a.localeCompare(b));
  return (
    <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
      {entries.map(([id, val]) => (
        <div key={id} style={{ display: "grid", gridTemplateColumns: "32px 1fr 48px", gap: 8, alignItems: "center" }}>
          <span style={{ fontWeight: 700, fontSize: 13, fontFamily: "var(--display)" }}>{id}</span>
          <div style={{ height: 6, borderRadius: 99, background: "rgba(255,255,255,.04)", overflow: "hidden" }}>
            <motion.div
              style={{ height: "100%", borderRadius: 99, background: trustColor(val) }}
              animate={{ width: `${Math.max(2, val * 100)}%` }}
              transition={{ type: "spring", stiffness: 200, damping: 20 }}
            />
          </div>
          <span style={{ fontFamily: "var(--mono)", fontSize: 12, fontWeight: 600, textAlign: "right", color: trustColor(val) }}>
            {val.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}

function PhaseResultCard({ result, variant }: { result: PhaseResult; variant: "bad" | "good" }) {
  return (
    <div className="jw-results">
      <div className="jw-stat">
        <div className="lbl">Score</div>
        <div className="val" style={{ color: variant === "bad" ? "var(--red)" : "var(--green)" }}>
          {result.score.toFixed(3)}
        </div>
      </div>
      <div className="jw-stat">
        <div className="lbl">Poisonings</div>
        <div className="val" style={{ color: result.poisonings > 0 ? "var(--red)" : "var(--ink)" }}>
          {result.poisonings}
        </div>
      </div>
      <div className="jw-stat">
        <div className="lbl">Detections</div>
        <div className="val" style={{ color: result.detections > 0 ? "var(--green)" : "var(--ink3)" }}>
          {result.detections}
        </div>
      </div>
    </div>
  );
}
