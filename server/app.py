"""
MoodMap Passive Mental Health — OpenEnv REST API (server/app.py)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

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

VALID_TASKS = ["triage", "risk_stratification", "early_warning"]
VALID_DIFFICULTIES = ["easy", "medium", "hard"]

# Must match graders/__init__.py and openenv.yaml score_range exactly
MIN_SCORE = 0.06
MAX_SCORE = 0.94

sessions: dict[str, MoodMapEnv] = {}


def _clamp(v: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, float(v)))


@app.get("/")
def home():
    try:
        dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard.html")
        with open(dashboard_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {"message": "MoodMap API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "moodmap-openenv", "version": "1.0.0"}


@app.get("/info")
def info():
    return {
        "name": "MoodMap Passive Mental Health Environment",
        "tasks": VALID_TASKS,
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


@app.get("/metadata")
def metadata():
    return {
        "name": "MoodMap Passive Mental Health Environment",
        "description": (
            "A passive mental health monitoring environment where an LLM agent analyzes "
            "behavioral signals to assess patient risk and recommend interventions."
        ),
        "tasks": VALID_TASKS,
        "difficulties": VALID_DIFFICULTIES,
        "reward_range": [MIN_SCORE, MAX_SCORE],
    }


@app.post("/reset")
def reset(body: Optional[Dict[str, Any]] = Body(default=None)):
    body = body or {}
    task = str(body.get("task", "triage")).lower().strip()
    difficulty = str(body.get("difficulty", "easy")).lower().strip()
    max_steps = int(body.get("max_steps", 5))

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
    max_steps = max(1, min(20, max_steps))

    env = MoodMapEnv(task=task, difficulty=difficulty, max_steps=max_steps)
    state = env.reset()
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    sessions[session_id] = env

    return {"session_id": session_id, **state}


@app.post("/step")
def step(body: Dict[str, Any]):
    session_id = body.get("session_id")
    if not session_id or session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )

    env = sessions[session_id]

    risk_score = _clamp(float(body.get("risk_score", 0.5)))
    confidence = _clamp(float(body.get("confidence", 0.5)))

    action = AgentAction(
        patient_id=str(body.get("patient_id", "unknown")),
        risk_score=risk_score,
        recommended_intervention=str(body.get("recommended_intervention", "no_action")),
        urgency_level=str(body.get("urgency_level", "medium")),
        reasoning=str(body.get("reasoning", "")),
        confidence=confidence,
    )

    result = env.step(action)

    grader_score: float = grade(
        task=env.task,
        action=action.model_dump(),
        ground_truth={
            "true_risk": result["reward_breakdown"].get("true_risk", risk_score),
            "ideal_intervention": result["reward_breakdown"].get(
                "ideal_intervention", action.recommended_intervention
            ),
        },
    )

    result["grader_score"] = _clamp(float(grader_score))
    result["reward"] = _clamp(float(result["reward"]))

    if result["done"]:
        result["episode_summary"] = env.get_episode_summary()
        del sessions[session_id]

    return result


@app.get("/sessions")
def list_sessions():
    return {
        "active_sessions": len(sessions),
        "session_ids": list(sessions.keys()),
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()