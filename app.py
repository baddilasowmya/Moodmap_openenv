"""
MoodMap Passive Mental Health — OpenEnv REST API

Endpoints follow the OpenEnv spec:
  POST /reset  → start a new episode
  POST /step   → submit an action, get reward
  GET  /health → health check
  GET  /info   → environment metadata
  GET  /       → dashboard HTML
"""

import os
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

VALID_TASKS        = ["triage", "risk_stratification", "early_warning"]
VALID_DIFFICULTIES = ["easy", "medium", "hard"]

# Hard score bounds — strictly open interval, never 0.0 or 1.0
MIN_SCORE = 0.05
MAX_SCORE = 0.95


def _clamp_score(v: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, float(v)))


class ResetRequest(BaseModel):
    task:       Optional[str] = Field("triage", description="One of: triage, risk_stratification, early_warning")
    difficulty: Optional[str] = Field("easy",   description="One of: easy, medium, hard")
    max_steps:  Optional[int] = Field(5, ge=1, le=20)


class StepRequest(BaseModel):
    session_id:               str
    patient_id:               str
    risk_score:               float = Field(..., ge=0.0, le=1.0)
    recommended_intervention: str
    urgency_level:            str
    reasoning:                str
    confidence:               float = Field(..., ge=0.0, le=1.0)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the live ICU-style dashboard."""
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
        "name":      "MoodMap Passive Mental Health Environment",
        "tasks":      VALID_TASKS,
        "difficulties": VALID_DIFFICULTIES,
        "reward_range": [MIN_SCORE, MAX_SCORE],
        "reward_components": ["detection", "intervention", "timeliness", "harm_avoidance"],
        "observation_space": {
            "type": "structured",
            "fields": [
                "patient_id", "age", "gender", "baseline_mood_score",
                "signals.sleep_hours", "signals.activity_steps",
                "signals.screen_time_hours", "signals.social_interactions",
                "signals.heart_rate_variability", "signals.app_usage_variance",
                "signals.typing_speed_change", "signals.location_entropy",
                "history_days", "prior_episodes", "medication_adherence",
            ],
        },
        "action_space": {
            "type": "structured",
            "fields": [
                "risk_score", "recommended_intervention",
                "urgency_level", "reasoning", "confidence",
            ],
        },
    }


@app.post("/reset")
async def reset(req: Optional[ResetRequest] = Body(default=None)):
    # If no body was sent, use all defaults
    if req is None:
        req = ResetRequest()

    # Normalise and validate task / difficulty
    task       = (req.task or "triage").lower().strip()
    difficulty = (req.difficulty or "easy").lower().strip()
    max_steps  = req.max_steps if req.max_steps is not None else 5

    if task not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Must be one of {VALID_TASKS}.",
        )
    if difficulty not in VALID_DIFFICULTIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid difficulty '{difficulty}'. Must be one of {VALID_DIFFICULTIES}.",
        )

    env = MoodMapEnv(task=task, difficulty=difficulty, max_steps=max_steps)
    state = env.reset()
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    _sessions[session_id] = env
    return {"session_id": session_id, **state}


@app.post("/step")
async def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found. Call /reset first.",
        )

    # Clamp risk_score and confidence to (0.05, 0.95) before building the action
    safe_risk_score = _clamp_score(req.risk_score)
    safe_confidence = _clamp_score(req.confidence)

    action = AgentAction(
        patient_id=req.patient_id,
        risk_score=safe_risk_score,
        recommended_intervention=req.recommended_intervention,
        urgency_level=req.urgency_level,
        reasoning=req.reasoning,
        confidence=safe_confidence,
    )

    result = env.step(action)

    # Run task-specific grader — grade() returns a float in (0.05, 0.95)
    grader_score: float = grade(
        task=env.task,
        action=action.model_dump(),
        ground_truth={
            "true_risk":          result["reward_breakdown"].get("true_risk", safe_risk_score),
            "ideal_intervention": result["reward_breakdown"].get("ideal_intervention", req.recommended_intervention),
        },
    )

    # Safety clamp on grader output (should already be clamped, but be defensive)
    result["grader_score"] = _clamp_score(grader_score)

    # Final safety clamp on main reward
    result["reward"] = _clamp_score(result["reward"])

    if result["done"]:
        result["episode_summary"] = env.get_episode_summary()
        del _sessions[req.session_id]

    return result


@app.get("/sessions")
async def list_sessions():
    return {
        "active_sessions": len(_sessions),
        "session_ids":     list(_sessions.keys()),
    }