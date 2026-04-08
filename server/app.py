"""
MoodMap Passive Mental Health — OpenEnv REST API
Endpoints follow the OpenEnv spec:
  POST /reset   → start a new episode
  POST /step    → submit an action, get reward
  GET  /health  → health check
  GET  /        → dashboard HTML
"""
import os
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from moodmap_env import MoodMapEnv
from moodmap_env.models import AgentAction
from graders import grade

app = FastAPI(
    title="MoodMap Passive Mental Health Environment",
    description="An RL environment for passive mental health monitoring via behavioral signals.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (stateless per Hugging Face Spaces restart)
_sessions: dict[str, MoodMapEnv] = {}


class ResetRequest(BaseModel):
    task: str = Field("triage", description="One of: triage, risk_stratification, early_warning")
    difficulty: str = Field("easy", description="One of: easy, medium, hard")
    max_steps: int = Field(5, ge=1, le=20)

    model_config = {"extra": "ignore"}


class StepRequest(BaseModel):
    session_id: str
    patient_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    recommended_intervention: str
    urgency_level: str
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the live ICU-style dashboard"""
    try:
        with open("dashboard.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>MoodMap API is running. Dashboard not found.</h1>")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "moodmap-openenv", "version": "1.0.0"}


@app.get("/info")
async def info():
    return {
        "name": "MoodMap Passive Mental Health Environment",
        "tasks": ["triage", "risk_stratification", "early_warning"],
        "difficulties": ["easy", "medium", "hard"],
        "reward_range": [0.05, 0.95],
        "reward_components": ["detection", "intervention", "timeliness", "harm_avoidance"],
        "observation_space": {
            "type": "structured",
            "fields": [
                "patient_id", "age", "gender", "baseline_mood_score",
                "signals.sleep_hours", "signals.activity_steps",
                "signals.screen_time_hours", "signals.social_interactions",
                "signals.heart_rate_variability", "signals.app_usage_variance",
                "signals.typing_speed_change", "signals.location_entropy",
                "history_days", "prior_episodes", "medication_adherence"
            ]
        },
        "action_space": {
            "type": "structured",
            "fields": ["risk_score", "recommended_intervention", "urgency_level", "reasoning", "confidence"]
        }
    }


@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None):
    import uuid
    if req is None:
        req = ResetRequest()
    env = MoodMapEnv(task=req.task, difficulty=req.difficulty, max_steps=req.max_steps)
    state = env.reset()
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    _sessions[session_id] = env
    return {"session_id": session_id, **state}


@app.post("/step")
async def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found. Call /reset first.")

    action = AgentAction(
        patient_id=req.patient_id,
        risk_score=req.risk_score,
        recommended_intervention=req.recommended_intervention,
        urgency_level=req.urgency_level,
        reasoning=req.reasoning,
        confidence=req.confidence,
    )

    result = env.step(action)

    # Also run task-specific grader
    grader_result = grade(
        task=env.task,
        action=action.model_dump(),
        ground_truth={
            "true_risk": result["reward_breakdown"].get("true_risk", req.risk_score),
            "ideal_intervention": result["reward_breakdown"].get("ideal_intervention", req.recommended_intervention),
        }
    )
    result["grader_score"] = grader_result

    if result["done"]:
        result["episode_summary"] = env.get_episode_summary()
        del _sessions[req.session_id]

    return result


@app.get("/sessions")
async def list_sessions():
    return {"active_sessions": len(_sessions), "session_ids": list(_sessions.keys())}
