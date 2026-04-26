"use client";
import SpecialistNetwork from "./SpecialistNetwork";
import TrustTimeline from "./TrustTimeline";
import MissionBriefing from "./MissionBriefing";
import ActionCenter from "./ActionCenter";
import FlightRecorder from "./FlightRecorder";
import type { ActionType, AutoPolicy, Observation, EventItem } from "../lib/types";

export default function MissionControl({
  observation,
  trustDeltas,
  activeSpec,
  recommended,
  score,
  detections,
  poisonings,
  events,
  running,
  done,
  lastReq,
  lastRes,
  onStep,
  onAutoRun,
  onStop,
}: {
  observation: Observation | null;
  trustDeltas: Record<string, number>;
  activeSpec: string | null;
  recommended: { action: ActionType; specialist: string; trust: number };
  score: number | undefined;
  detections?: number;
  poisonings?: number;
  events: EventItem[];
  running: boolean;
  done: boolean;
  lastReq: Record<string, unknown> | null;
  lastRes: Record<string, unknown> | null;
  onStep: (action: ActionType) => void;
  onAutoRun: (policy: AutoPolicy) => void;
  onStop: () => void;
}) {
  return (
    <div className="mc">
      {/* left column */}
      <div className="mc-left">
        <div className="panel">
          <div className="panel-head">
            <div className="panel-eyebrow">Specialist Network</div>
            <div className="panel-title">Public Slots vs Hidden Risk</div>
          </div>
          <SpecialistNetwork
            observation={observation}
            trustDeltas={trustDeltas}
            activeSpec={activeSpec}
          />
        </div>

        <div className="panel">
          <div className="panel-head">
            <div className="panel-eyebrow">Trust Timeline</div>
            <div className="panel-title">Behavioral Trust Scores</div>
          </div>
          <TrustTimeline observation={observation} trustDeltas={trustDeltas} />
        </div>
      </div>

      {/* right column */}
      <div className="mc-right">
        <div className="panel">
          <div className="panel-head">
            <div className="panel-eyebrow">Mission Briefing</div>
            <div className="panel-title">Live Orchestrator State</div>
          </div>
          <MissionBriefing
            observation={observation}
            score={score}
            detections={detections}
            poisonings={poisonings}
          />
        </div>

        <div className="panel">
          <div className="panel-head">
            <div className="panel-eyebrow">Command</div>
            <div className="panel-title">Run the Orchestrator</div>
          </div>
          <ActionCenter
            recommended={recommended}
            running={running}
            done={done}
            onStep={onStep}
            onAutoRun={onAutoRun}
            onStop={onStop}
          />
        </div>

        <div className="panel">
          <div className="panel-head">
            <div className="panel-eyebrow">Flight Recorder</div>
            <div className="panel-title">Event Trail</div>
          </div>
          <FlightRecorder events={events} lastReq={lastReq} lastRes={lastRes} />
        </div>
      </div>
    </div>
  );
}
