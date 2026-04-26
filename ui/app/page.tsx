"use client";

import { useEffect } from "react";
import { useSentinel } from "./hooks/useSentinel";
import Landing from "./components/Landing";
import MissionControl from "./components/MissionControl";
import JudgeWizard from "./components/JudgeWizard";
import type { ViewMode, TaskType } from "./lib/types";

const TABS: { id: ViewMode; label: string }[] = [
  { id: "landing", label: "Overview" },
  { id: "mission", label: "Mission Control" },
  { id: "judge",   label: "Judge Demo" },
];

const TASKS: { value: TaskType; label: string }[] = [
  { value: "task1", label: "Task 1" },
  { value: "task2", label: "Task 2" },
  { value: "task3", label: "Task 3" },
];

export default function Page() {
  const s = useSentinel();

  /* auto-reset on first mount */
  useEffect(() => {
    void s.resetEpisode();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="shell">
      <div className="grid-bg" />

      {/* header */}
      <header className="hdr">
        <span className="hdr-brand">SENTINEL</span>
        <nav className="hdr-nav">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={s.view === t.id ? "on" : ""}
              onClick={() => s.setView(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div className="hdr-controls">
          <div className="hdr-pill">
            <label>Task</label>
            <select
              value={s.taskType}
              onChange={(e) => {
                const next = e.target.value as TaskType;
                s.setTaskType(next);
                void s.resetEpisode(next, s.seed);
              }}
            >
              {TASKS.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </div>
          <div className="hdr-pill">
            <label>Seed</label>
            <input
              type="number"
              value={s.seed}
              onChange={(e) => s.setSeed(Number(e.target.value || 0))}
            />
          </div>
          <button className="btn" onClick={() => void s.resetEpisode()}>
            Reset
          </button>
          <button className="btn btn-ghost" onClick={() => void s.swapProfiles()}>
            Swap Profiles
          </button>
        </div>
      </header>

      {/* content */}
      <main className="main">
        {s.view === "landing" && (
          <Landing
            proof={s.proof}
            onEnterMission={() => s.setView("mission")}
            onEnterJudge={() => s.setView("judge")}
          />
        )}

        {s.view === "mission" && (
          <MissionControl
            observation={s.observation}
            trustDeltas={s.trustDeltas}
            activeSpec={s.activeSpec}
            recommended={s.recommended}
            score={s.info?.score}
            detections={s.info?.adversarial_detections}
            poisonings={s.info?.adversarial_poisonings}
            events={s.events}
            running={s.running}
            done={s.done}
            lastReq={s.lastReq}
            lastRes={s.lastRes}
            onStep={(action) => void s.stepEpisode(action)}
            onAutoRun={(policy) => void s.autoRun(policy)}
            onStop={s.stopAutoRun}
          />
        )}

        {s.view === "judge" && (
          <JudgeWizard
            autoRun={s.autoRun}
            resetEpisode={s.resetEpisode}
            swapProfiles={s.swapProfiles}
            observation={s.observation}
            events={s.events}
            info={s.info}
            running={s.running}
          />
        )}
      </main>
    </div>
  );
}
