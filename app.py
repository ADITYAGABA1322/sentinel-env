from __future__ import annotations

import asyncio
import html
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from difficulty_controller import GLOBAL_DIFFICULTY_CONTROLLER
from environment import SentinelEnv
from mission_context import build_orchestrator_prompt, mission_for_task, problem_statement
from scenarios import scenario_summary
from sentinel_config import SESSION_BACKEND, SESSION_MAX_ACTIVE, SESSION_TTL_SECONDS

# ---------------------------------------------------------------------------
# App + session store
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SENTINEL — Multi-Agent Trust Calibration Environment",
    description=(
        "OpenEnv-compatible RL environment where an orchestrator agent learns "
        "dynamic trust calibration across adversarial long-horizon tasks."
    ),
    version="1.0.0",
)

@dataclass
class SessionEntry:
    env: SentinelEnv
    created_at: float
    last_access_at: float


class SessionStore:
    """
    Single-process TTL + LRU store for active SentinelEnv objects.

    This is intentionally memory-backed for OpenEnv/HF Space simplicity. It is
    safe for the Dockerfile's single-worker deployment. If you increase workers,
    use sticky routing or replace this with a shared backend such as Redis.
    """

    def __init__(self, ttl_seconds: int, max_active: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_active = max_active
        self._items: OrderedDict[str, SessionEntry] = OrderedDict()
        self._lock = RLock()

    def set(self, session_id: str, env: SentinelEnv) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            self._items[session_id] = SessionEntry(env=env, created_at=now, last_access_at=now)
            self._items.move_to_end(session_id)
            while len(self._items) > self._max_active:
                self._items.popitem(last=False)

    def get(self, session_id: str) -> SentinelEnv | None:
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            entry = self._items.get(session_id)
            if entry is None:
                return None
            entry.last_access_at = now
            self._items.move_to_end(session_id)
            return entry.env

    def pop(self, session_id: str) -> SentinelEnv | None:
        with self._lock:
            entry = self._items.pop(session_id, None)
            return entry.env if entry else None

    def stats(self) -> dict[str, int | str | bool]:
        with self._lock:
            self._prune_locked(time.monotonic())
            return {
                "backend": SESSION_BACKEND,
                "active_sessions": len(self._items),
                "ttl_seconds": self._ttl_seconds,
                "max_active": self._max_active,
                "multi_worker_safe": False,
            }

    def _prune_locked(self, now: float) -> None:
        expired = [
            sid
            for sid, entry in self._items.items()
            if now - entry.last_access_at > self._ttl_seconds
        ]
        for sid in expired:
            self._items.pop(sid, None)


_sessions = SessionStore(ttl_seconds=SESSION_TTL_SECONDS, max_active=SESSION_MAX_ACTIVE)
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
_FRONTEND_OUT_DIR = Path(__file__).resolve().parent / "ui" / "out"
_FRONTEND_NEXT_DIR = _FRONTEND_OUT_DIR / "_next"

if _FRONTEND_NEXT_DIR.exists():
    app.mount("/_next", StaticFiles(directory=_FRONTEND_NEXT_DIR), name="next-assets")

def _get_env(session_id: str) -> SentinelEnv:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return env


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_type:   str | None = None
    scenario_id: str | None = None
    seed:        int | None = None
    adaptive:    bool = False

class StepRequest(BaseModel):
    session_id:       str
    task_type:        str
    action_type:      str                  # delegate | verify | solve_independently | skip
    specialist_id:    str | None = None
    subtask_response: str | None = None
    reasoning:        str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": "sentinel-env",
        "version": "1.0.0",
        "session_store": _sessions.stats(),
    }


@app.get("/")
def root():
    frontend_index = _FRONTEND_OUT_DIR / "index.html"
    if frontend_index.exists():
        return FileResponse(frontend_index)
    index_path = _STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(
        {
            "name": "sentinel-env",
            "status": "ok",
            "summary": (
                "SENTINEL trains an orchestrator to calibrate trust, verify risky "
                "outputs, recover from failures, and finish long multi-agent tasks."
            ),
            "routes": [
                "/health", "/problem", "/mission", "/metadata", "/tasks", "/schema",
                "/grader", "/reward-report", "/difficulty", "/stream", "/trust-dashboard",
                "/reset", "/step", "/state",
            ],
        }
    )


@app.get("/assets/baseline_comparison.png")
def baseline_comparison_chart():
    chart_path = _OUTPUTS_DIR / "baseline_comparison.png"
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="Baseline comparison chart not found.")
    return FileResponse(chart_path, media_type="image/png")


@app.get("/assets/evaluation_results.json")
def evaluation_results():
    results_path = _OUTPUTS_DIR / "evaluation_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found.")
    return FileResponse(results_path, media_type="application/json")


@app.get("/api")
def api_root():
    return {
        "name": "sentinel-env",
        "status": "ok",
        "summary": (
            "SENTINEL trains an orchestrator to calibrate trust, verify risky "
            "outputs, recover from failures, and finish long multi-agent tasks."
        ),
        "routes": [
            "/health", "/problem", "/mission", "/metadata", "/tasks", "/schema",
            "/grader", "/reward-report", "/difficulty", "/stream", "/trust-dashboard",
            "/reset", "/step", "/state",
        ],
    }


@app.get("/problem")
def problem():
    """Judge-readable explanation of what the environment solves."""
    return problem_statement()


@app.get("/mission")
def mission(task_type: str = Query("task3", pattern="^task[123]$")):
    """Real-world wrapper for each abstract OpenEnv task."""
    return {
        "task_type": task_type,
        "mission": mission_for_task(task_type),
        "how_to_use": (
            "Call /reset to get an observation, then ask an orchestrator model to "
            "emit one JSON action for /step."
        ),
    }


@app.get("/metadata")
def metadata():
    summary = scenario_summary()
    return {
        "name":        "sentinel-env",
        "version":     "1.0.0",
        "description": "Multi-agent trust calibration RL environment.",
        "tasks": {
            "task1": {"name": "Single-Step Trust Decision", "difficulty": "easy",  "subtasks": 10, "max_steps": 15},
            "task2": {"name": "Multi-Step Delegation Chain","difficulty": "medium","subtasks": 15, "max_steps": 30},
            "task3": {"name": "Full Adversarial Episode",   "difficulty": "hard",  "subtasks": 20, "max_steps": 45},
        },
        "specialists": ["S0 (AccurateSlow)", "S1 (OverconfidentFast)",
                        "S2 (DomainBound)", "S3 (Adversarial)", "S4 (Degrading)"],
        "action_types": ["delegate", "verify", "solve_independently", "skip"],
        "scenarios": summary,
        "reward_range": "(0.01, 0.99) boundary-exclusive",
        "observation_features": [
            "trust_snapshot",
            "behavioral_fingerprints.confidence_accuracy_gap",
            "behavioral_fingerprints.domain_hit_rate",
            "behavioral_fingerprints.stakes_volatility",
            "difficulty_profile",
        ],
        "real_world_bridge": problem_statement()["problem"]["not_a_simple_prompt_solver"],
        "deployment_contract": {
            "session_backend": SESSION_BACKEND,
            "single_worker_required": True,
            "reason": "Active SentinelEnv objects live in one process memory with TTL/LRU cleanup.",
            "ttl_seconds": SESSION_TTL_SECONDS,
            "max_active_sessions": SESSION_MAX_ACTIVE,
        },
        "adaptive_curriculum": GLOBAL_DIFFICULTY_CONTROLLER.state(),
    }


@app.get("/tasks")
def tasks():
    return {
        "task1": {
            "difficulty": "easy",
            "description": "10-subtask linear chain. Choose which specialist to delegate each subtask to.",
            "adversary_active": False,
            "reward": "0.99 correct delegation + stakes awareness | 0.02 skip",
            "mission": mission_for_task("task1"),
        },
        "task2": {
            "difficulty": "medium",
            "description": "15-subtask branching DAG. Build trust profile across honest specialists.",
            "adversary_active": False,
            "reward": "0.99 completion × calibration bonus",
            "mission": mission_for_task("task2"),
        },
        "task3": {
            "difficulty": "hard",
            "description": "20-subtask full DAG. Adversarial specialist active. Detect and route around poison.",
            "adversary_active": True,
            "reward": "0.35×completion + 0.30×detection + 0.25×calibration + 0.10×efficiency",
            "mission": mission_for_task("task3"),
        },
    }


@app.get("/schema")
def schema():
    return {
        "reset_request":  ResetRequest.model_json_schema(),
        "step_request":   StepRequest.model_json_schema(),
    }


@app.get("/grader")
def grader():
    return {
        "task1": {
            "step":     "accuracy×0.43 + stakes×0.30 + efficiency×0.12 + confidence×0.07 + domain×0.04 + verify×0.04",
            "terminal": "same as last step",
        },
        "task2": {
            "step":     "accuracy×0.55 + efficiency×0.25 + confidence×0.10 + domain×0.10",
            "terminal": "completion_rate×0.65 + trust_calibration×0.35",
        },
        "task3": {
            "step":     "accuracy×0.32 + stakes×0.33 + efficiency×0.10 + confidence×0.10 + verify×0.10 + domain×0.05",
            "terminal": "completion×0.35 + detection×0.30 + calibration×0.25 + efficiency×0.10",
        },
    }


@app.get("/reward-report")
def reward_report(session_id: str = Query(...)):
    env = _get_env(session_id)
    return env.reward_report()


@app.get("/difficulty")
def difficulty():
    return {
        "controller": GLOBAL_DIFFICULTY_CONTROLLER.state(),
        "how_to_enable": "POST /reset with {\"task_type\":\"task3\",\"adaptive\":true}.",
    }


@app.post("/difficulty/reset")
def reset_difficulty():
    GLOBAL_DIFFICULTY_CONTROLLER.reset()
    return {"controller": GLOBAL_DIFFICULTY_CONTROLLER.state()}


@app.get("/stream")
async def stream(session_id: str = Query(...)):
    async def event_gen():
        while True:
            env = _sessions.get(session_id)
            if env is None:
                yield "event: close\ndata: {\"reason\":\"session_not_found\"}\n\n"
                break
            yield f"data: {json.dumps(env.stream_snapshot())}\n\n"
            if env.done:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/trust-dashboard")
def trust_dashboard(session_id: str = Query("")):
    return HTMLResponse(_trust_dashboard_html(session_id))


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    env = SentinelEnv()
    result = env.reset(
        task_type=req.task_type,
        scenario_id=req.scenario_id,
        seed=req.seed,
        adaptive=req.adaptive,
    )
    session_id = result["info"]["session_id"]
    _sessions.set(session_id, env)
    result["info"]["mission"] = mission_for_task(result["observation"]["task_type"])
    result["info"]["orchestrator_prompt"] = build_orchestrator_prompt(result["observation"])
    return result


@app.post("/step")
def step(req: StepRequest, session_id: str = Query(...)):
    env = _get_env(session_id)
    try:
        result = env.step(req.model_dump())
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Clean up completed sessions to avoid memory leak
    if result["done"]:
        _sessions.pop(session_id)
    else:
        result["info"]["orchestrator_prompt"] = build_orchestrator_prompt(result["observation"])

    return result


@app.get("/state")
def state(session_id: str = Query(...)):
    env = _get_env(session_id)
    return env.state(session_id=session_id)


@app.post("/mcp")
def mcp(body: dict[str, Any]):
    """MCP-compatible endpoint for tool-calling agents."""
    method = body.get("method", "")
    params = body.get("params", {})

    if method == "reset":
        env = SentinelEnv()
        result = env.reset(**params)
        session_id = result["info"]["session_id"]
        _sessions.set(session_id, env)
        result["info"]["mission"] = mission_for_task(result["observation"]["task_type"])
        result["info"]["orchestrator_prompt"] = build_orchestrator_prompt(result["observation"])
        return {"result": result}

    elif method == "step":
        session_id = params.get("session_id") or body.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required for step.")
        env = _get_env(session_id)
        result = env.step(params)
        if result["done"]:
            _sessions.pop(session_id)
        else:
            result["info"]["orchestrator_prompt"] = build_orchestrator_prompt(result["observation"])
        return {"result": result}

    elif method == "state":
        session_id = params.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required for state.")
        return {"result": _get_env(session_id).state(session_id)}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")


def _trust_dashboard_html(session_id: str) -> str:
    escaped_session = html.escape(session_id, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SENTINEL Trust Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0b0f14;
      color: #e5eef8;
    }}
    body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #0b0f14; }}
    main {{ width: min(1040px, calc(100vw - 32px)); }}
    header {{ display: flex; justify-content: space-between; gap: 24px; align-items: end; margin-bottom: 28px; }}
    h1 {{ margin: 0; font-size: clamp(28px, 5vw, 56px); letter-spacing: 0; }}
    p {{ color: #94a3b8; line-height: 1.6; margin: 8px 0 0; max-width: 640px; }}
    input {{ width: 360px; max-width: 100%; background: #111827; color: #e5eef8; border: 1px solid #263241; border-radius: 8px; padding: 11px 12px; }}
    button {{ background: #e5eef8; color: #0b0f14; border: 0; border-radius: 8px; padding: 11px 14px; font-weight: 700; cursor: pointer; }}
    .controls {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: end; }}
    .panel {{ border: 1px solid #223043; background: #0f1722; border-radius: 8px; padding: 24px; box-shadow: 0 24px 80px rgba(0,0,0,.32); }}
    .bar {{ display: grid; grid-template-columns: 56px 1fr 74px; align-items: center; gap: 16px; margin: 18px 0; }}
    .id {{ font-weight: 800; font-size: 22px; }}
    .track {{ height: 28px; background: #182231; border-radius: 6px; overflow: hidden; border: 1px solid #263241; }}
    .fill {{ height: 100%; width: 50%; background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981); transition: width .35s ease; }}
    .score {{ font-variant-numeric: tabular-nums; text-align: right; color: #d9f99d; font-size: 22px; font-weight: 800; }}
    .meta {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 22px; }}
    .stat {{ border: 1px solid #223043; background: #0b111a; border-radius: 8px; padding: 14px; }}
    .label {{ color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .value {{ margin-top: 8px; font-size: 18px; font-weight: 800; }}
    @media (max-width: 760px) {{
      header, .meta {{ display: block; }}
      .controls {{ justify-content: stretch; margin-top: 18px; }}
      input, button {{ width: 100%; }}
      .stat {{ margin-top: 12px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>SENTINEL Live Trust</h1>
        <p>Watch the orchestrator's trust ledger move in real time as specialists prove reliable, degrade, or get caught poisoning high-stakes work.</p>
      </div>
      <div class="controls">
        <input id="sid" placeholder="session_id" value="{escaped_session}" />
        <button onclick="connect()">Connect</button>
      </div>
    </header>
    <section class="panel" id="bars"></section>
  </main>
  <script>
    const ids = ["S0", "S1", "S2", "S3", "S4"];
    const bars = document.getElementById("bars");
    bars.innerHTML = ids.map(id => `
      <div class="bar">
        <div class="id">${{id}}</div>
        <div class="track"><div class="fill" id="fill-${{id}}"></div></div>
        <div class="score" id="score-${{id}}">0.500</div>
      </div>
    `).join("") + `
      <div class="meta">
        <div class="stat"><div class="label">step</div><div class="value" id="step">0 / 0</div></div>
        <div class="stat"><div class="label">last reward</div><div class="value" id="reward">0.000</div></div>
        <div class="stat"><div class="label">adaptive threshold</div><div class="value" id="threshold">0.700</div></div>
      </div>`;
    let source = null;
    function connect() {{
      if (source) source.close();
      const sid = document.getElementById("sid").value.trim();
      if (!sid) return;
      source = new EventSource(`/stream?session_id=${{encodeURIComponent(sid)}}`);
      source.onmessage = event => {{
        const data = JSON.parse(event.data);
        ids.forEach(id => {{
          const value = data.trust_snapshot?.[id] ?? 0.5;
          document.getElementById(`fill-${{id}}`).style.width = `${{Math.round(value * 100)}}%`;
          document.getElementById(`score-${{id}}`).textContent = Number(value).toFixed(3);
        }});
        document.getElementById("step").textContent = `${{data.step_count}} / ${{data.max_steps}}`;
        document.getElementById("reward").textContent = Number(data.last_reward || 0).toFixed(3);
        document.getElementById("threshold").textContent = Number(data.difficulty_profile?.adversarial_threshold || 0.7).toFixed(3);
      }};
    }}
    if (document.getElementById("sid").value.trim()) connect();
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
