"use client";

import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import type {
  ViewMode, TaskType, ActionType, AutoPolicy,
  StepResult, EventItem, EvaluationData,
  Observation,
} from "../lib/types";

/* ── helpers ──────────────────────────────────────────── */

function bestSpec(obs: Observation | null): string {
  if (!obs) return "S0";
  return [...obs.available_specialists].sort(
    (a, b) => (obs.trust_snapshot[b] ?? 0.5) - (obs.trust_snapshot[a] ?? 0.5),
  )[0];
}

function heuristicMove(obs: Observation | null) {
  if (!obs) return { action: "delegate" as ActionType, specialist: "S0", trust: 0.5 };
  const sp = bestSpec(obs);
  const t  = obs.trust_snapshot[sp] ?? 0.5;
  if (obs.stakes_level >= 0.7 && t < 0.65)
    return { action: "verify" as ActionType, specialist: sp, trust: t };
  return { action: "delegate" as ActionType, specialist: sp, trust: t };
}

function randomMove(obs: Observation | null) {
  if (!obs) return { action: "delegate" as ActionType, specialist: "S0", trust: 0.5 };
  const sp = obs.available_specialists[
    Math.floor(Math.random() * obs.available_specialists.length)
  ] || "S0";
  return { action: "delegate" as ActionType, specialist: sp, trust: obs.trust_snapshot[sp] ?? 0.5 };
}

type ReplayRow = {
  task_type: TaskType;
  seed: number;
  step: number;
  action: {
    action_type: ActionType;
    specialist_id?: string | null;
    reasoning?: string;
  };
};

function replayMove(
  obs: Observation | null,
  seed: number,
  replay: Map<string, ReplayRow>,
) {
  if (!obs) return heuristicMove(obs);
  const key = `${obs.task_type}:${seed}:${obs.step_count}`;
  const row = replay.get(key);
  if (!row) return { ...heuristicMove(obs), replayMiss: true };
  const action = row.action.action_type;
  const specialist = row.action.specialist_id || bestSpec(obs);
  return {
    action,
    specialist,
    trust: obs.trust_snapshot[specialist] ?? 0.5,
    replayMiss: false,
  };
}

function outcomeOf(reason: string): EventItem["outcome"] {
  const r = reason.toLowerCase();
  if (r.includes("poison") || r.includes("adversarial")) return "poisoned";
  if (r.includes("block") || r.includes("verif"))         return "blocked";
  if (r.includes("skip"))                                  return "skipped";
  return "success";
}

/* ── hook ─────────────────────────────────────────────── */

export function useSentinel() {
  const [view, setView]           = useState<ViewMode>("landing");
  const [taskType, setTaskType]   = useState<TaskType>("task3");
  const [seed, setSeed]           = useState(42);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [result, setResult]       = useState<StepResult | null>(null);
  const [running, setRunning]     = useState(false);
  const [events, setEvents]       = useState<EventItem[]>([]);
  const [lastReq, setLastReq]     = useState<Record<string, unknown> | null>(null);
  const [lastRes, setLastRes]     = useState<Record<string, unknown> | null>(null);
  const [evaluation, setEval]     = useState<EvaluationData | null>(null);
  const [replay, setReplay]       = useState<Map<string, ReplayRow>>(new Map());
  const [prevTrust, setPrevTrust] = useState<Record<string, number>>({});
  const [activeSpec, setActiveSpec] = useState<string | null>(null);

  const abortRef = useRef(false);

  /* load evaluation data once */
  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/assets/evaluation_results.json`)
      .then((r) => r.json())
      .then(setEval)
      .catch(() => null);

    fetch(`${process.env.NEXT_PUBLIC_API_URL}/assets/trained_policy_replay.jsonl`)
      .then((r) => r.ok ? r.text() : "")
      .then((txt) => {
        const table = new Map<string, ReplayRow>();
        for (const line of txt.split("\n")) {
          if (!line.trim()) continue;
          const row = JSON.parse(line) as ReplayRow;
          table.set(`${row.task_type}:${row.seed}:${row.step}`, row);
        }
        setReplay(table);
      })
      .catch(() => null);
  }, []);

  const observation = result?.observation ?? null;
  const info        = result?.info;
  const reward      = result?.reward;
  const done        = result?.done ?? false;

  /* trust deltas */
  const trustDeltas = useMemo(() => {
    if (!observation) return {};
    const d: Record<string, number> = {};
    for (const id of observation.available_specialists) {
      d[id] = (observation.trust_snapshot[id] ?? 0.5) - (prevTrust[id] ?? 0.5);
    }
    return d;
  }, [observation, prevTrust]);

  const recommended = useMemo(() => heuristicMove(observation), [observation]);

  const proof = useMemo(() => {
    if (!evaluation) return null;
    return {
      random:         evaluation.summary.random,
      heuristic:      evaluation.summary.heuristic,
      oracle:         evaluation.summary.oracle_lite,
      trained:        evaluation.summary.trained,
      task3Random:    evaluation.by_task.task3.random,
      task3Heuristic: evaluation.by_task.task3.heuristic,
    };
  }, [evaluation]);

  /* ── API calls ──────────────────────────────────────── */

  const resetEpisode = useCallback(
    async (nextTask?: TaskType, nextSeed?: number): Promise<StepResult | null> => {
      const t = nextTask ?? taskType;
      const s = nextSeed ?? seed;
      setRunning(true);
      abortRef.current = false;
      const payload = { task_type: t, seed: s };
      setLastReq({ method: "POST", path: "/reset", body: payload });
      try {
        const res  = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = (await res.json()) as StepResult;
        setPrevTrust({});
        setResult(data);
        setLastRes(data as unknown as Record<string, unknown>);
        setSessionId(data.info.session_id);
        setActiveSpec(null);
        setEvents([{
          step: 0, action: "reset", summary: "Episode initialized. Hidden profiles reshuffled.",
          reward: 0, outcome: "reset",
        }]);
        return data;
      } finally {
        setRunning(false);
      }
    },
    [taskType, seed],
  );

  const stepEpisode = useCallback(
    async (
      action: ActionType,
      specialistOverride?: string,
      ctx?: StepResult | null,
    ): Promise<StepResult | null> => {
      const active   = ctx ?? result;
      const obs      = active?.observation ?? observation;
      const sid      = active?.info.session_id ?? sessionId;
      if (!sid || !obs || running || active?.done) return null;

      setRunning(true);
      const specialist =
        action === "delegate" || action === "verify"
          ? specialistOverride || bestSpec(obs)
          : null;

      setActiveSpec(specialist);

      const payload = {
        session_id: sid,
        task_type: obs.task_type,
        action_type: action,
        specialist_id: specialist,
        subtask_response: action === "solve_independently" ? "SELF_SOLVED" : null,
        reasoning: `ui-${action}${specialist ? `-${specialist}` : ""}`,
      };
      setLastReq({ method: "POST", path: `/step?session_id=${sid}`, body: payload });
      try {
        const res  = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/step?session_id=${encodeURIComponent(sid)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = (await res.json()) as StepResult;
        setPrevTrust(obs.trust_snapshot);
        setResult(data);
        setLastRes(data as unknown as Record<string, unknown>);
        setEvents((prev) => [
          ...prev,
          {
            step: data.info.step_count,
            action: action,
            specialist: specialist ?? undefined,
            summary: data.reward.reason,
            reward: data.reward.value,
            outcome: outcomeOf(data.reward.reason),
          },
        ]);
        return data;
      } finally {
        setRunning(false);
      }
    },
    [result, observation, sessionId, running],
  );

  const autoRun = useCallback(
    async (policy: AutoPolicy) => {
      abortRef.current = false;
      let local = result;
      if (!local || local.done) {
        local = await resetEpisode();
      }
      let guard = 0;
      while (!local?.done && guard < 70 && !abortRef.current) {
        const obs = local?.observation ?? null;
        const mv =
          policy === "random"
            ? randomMove(obs)
            : policy === "trained"
              ? replayMove(obs, seed, replay)
              : heuristicMove(obs);
        const next = await stepEpisode(mv.action, mv.specialist, local);
        guard += 1;
        await new Promise((r) => setTimeout(r, 140));
        if (!next) break;
        local = next;
      }
    },
    [result, resetEpisode, stepEpisode, replay, seed],
  );

  const stopAutoRun = useCallback(() => { abortRef.current = true; }, []);

  const swapProfiles = useCallback(async () => {
    const ns = seed + 1;
    setSeed(ns);
    await resetEpisode(taskType, ns);
  }, [seed, taskType, resetEpisode]);

  return {
    view, setView,
    taskType, setTaskType,
    seed, setSeed,
    sessionId, observation, info, reward, result, running, done,
    events, lastReq, lastRes, evaluation, proof,
    prevTrust, trustDeltas, recommended, activeSpec,
    resetEpisode, stepEpisode, autoRun, stopAutoRun, swapProfiles,
  };
}
