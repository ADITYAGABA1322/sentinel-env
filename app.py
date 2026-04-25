from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from environment import SentinelEnv
from scenarios import scenario_summary

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

# One env instance per session_id
_sessions: dict[str, SentinelEnv] = {}

def _get_env(session_id: str) -> SentinelEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_type:   str | None = None
    scenario_id: str | None = None
    seed:        int | None = None

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
    return {"status": "ok", "environment": "sentinel-env", "version": "1.0.0"}


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
    }


@app.get("/tasks")
def tasks():
    return {
        "task1": {
            "difficulty": "easy",
            "description": "10-subtask linear chain. Choose which specialist to delegate each subtask to.",
            "adversary_active": False,
            "reward": "0.99 correct delegation + stakes awareness | 0.02 skip",
        },
        "task2": {
            "difficulty": "medium",
            "description": "15-subtask branching DAG. Build trust profile across honest specialists.",
            "adversary_active": False,
            "reward": "0.99 completion × calibration bonus",
        },
        "task3": {
            "difficulty": "hard",
            "description": "20-subtask full DAG. Adversarial specialist active. Detect and route around poison.",
            "adversary_active": True,
            "reward": "0.35×completion + 0.30×detection + 0.25×calibration + 0.10×efficiency",
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
            "step":     "task_accuracy×0.50 + stakes_awareness×0.35 + efficiency×0.15",
            "terminal": "same as last step",
        },
        "task2": {
            "step":     "task_accuracy×0.65 + efficiency×0.35",
            "terminal": "completion_rate×0.65 + trust_calibration×0.35",
        },
        "task3": {
            "step":     "task_accuracy×0.40 + stakes_awareness×0.45 + efficiency×0.15",
            "terminal": "completion×0.35 + detection×0.30 + calibration×0.25 + efficiency×0.10",
        },
    }


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    env = SentinelEnv()
    result = env.reset(
        task_type=req.task_type,
        scenario_id=req.scenario_id,
        seed=req.seed,
    )
    session_id = result["info"]["session_id"]
    _sessions[session_id] = env
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
        _sessions.pop(session_id, None)

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
        _sessions[session_id] = env
        return {"result": result}

    elif method == "step":
        session_id = params.get("session_id") or body.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required for step.")
        env = _get_env(session_id)
        result = env.step(params)
        if result["done"]:
            _sessions.pop(session_id, None)
        return {"result": result}

    elif method == "state":
        session_id = params.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required for state.")
        return {"result": _get_env(session_id).state(session_id)}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Streaming (SSE)
# ---------------------------------------------------------------------------

@app.get("/stream")
async def stream_trust(session_id: str):
    """
    SSE endpoint. Streams trust state every 500ms for a given session.
    Client connects once and receives live updates until session ends.
    """

    async def event_generator():
        last_step = -1
        while True:
            if session_id not in _sessions:
                yield f"data: {json.dumps({'done': True})}\n\n"
                break

            env = _sessions[session_id]

            if env.step_count != last_step:
                last_step = env.step_count
                atlas = getattr(env, "atlas", None)
                payload = {
                    "step": env.step_count,
                    "trust": env._ledger.snapshot(),
                    "fingerprints": env._compute_live_fingerprints(),
                    "atlas_summary": atlas.get_summary() if atlas is not None else {"episodes_seen": 0, "archetypes_discovered": []},
                    "done": env.done,
                    "last_action": env.last_action_summary,
                    "episode_reward": round(env.total_reward, 3),
                }
                yield f"data: {json.dumps(payload)}\n\n"

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)