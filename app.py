"""
MoodMap Passive Mental Health — FastAPI Server
===============================================
POST /reset     — start a new episode (task: triage|risk_stratification|early_warning)
POST /step      — submit an agent action, get observation + reward
GET  /state     — current episode metadata
GET  /health    — health check
GET  /tasks     — list all 3 tasks + action schema
GET  /grader    — score for current episode strictly in (0,1)
GET  /baseline  — run baseline agent on all 3 tasks

Run:  uvicorn app:app --host 0.0.0.0 --port 7860 --reload
Docs: http://localhost:7860/docs
"""

import sys, os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict

from moodmap_env import MoodMapEnv
from moodmap_env.models import AgentAction
from graders import grade_triage, grade_risk_stratification, grade_early_warning

app = FastAPI(
    title="MoodMap Passive Mental Health Agent",
    description=(
        "LLM agent analyzes passive behavioral signals to assess patient mental "
        "health risk and recommend appropriate interventions."
    ),
    version="1.0.0",
)

MIN_SCORE = 0.06
MAX_SCORE = 0.94

def _clamp(v: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, float(v)))

# ── Global episode state ──────────────────────────────────────────────────────
_env: Optional[MoodMapEnv] = None
_task: str = "triage"
_difficulty: str = "easy"
_episode_reward_sum: float = 0.0
_episode_steps: int = 0
_last_grader_score: float = 0.5
_current_ground_truth: Optional[Dict] = None
_episode_done: bool = False

# ── Request models ────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task: Optional[str] = None
    difficulty: Optional[str] = None
    seed: Optional[int] = None

class StepRequest(BaseModel):
    patient_id: str = Field(default="unknown")
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    recommended_intervention: str = Field(default="no_action")
    urgency_level: str = Field(default="medium")
    reasoning: str = Field(default="")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _grade(task: str, action_dict: dict, ground_truth: dict) -> float:
    if task == "triage":
        return _clamp(grade_triage(action_dict, ground_truth))
    elif task == "risk_stratification":
        return _clamp(grade_risk_stratification(action_dict, ground_truth))
    elif task == "early_warning":
        return _clamp(grade_early_warning(action_dict, ground_truth))
    return 0.5

# ── Standard endpoints ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "project": "MoodMap Passive Mental Health Agent"}

@app.post("/reset")
async def reset(req: ResetRequest = None):
    """Start a new episode. task: triage | risk_stratification | early_warning"""
    global _env, _task, _difficulty, _episode_reward_sum, _episode_steps
    global _last_grader_score, _current_ground_truth, _episode_done

    task = (req.task if req else None) or "triage"
    diff = (req.difficulty if req else None) or "easy"

    if task not in ["triage", "risk_stratification", "early_warning"]:
        task = "triage"
    if diff not in ["easy", "medium", "hard"]:
        diff = "easy"

    _task = task
    _difficulty = diff
    _episode_reward_sum = 0.0
    _episode_steps = 0
    _last_grader_score = 0.5
    _episode_done = False

    _env = MoodMapEnv(task=task, difficulty=diff, max_steps=5)
    result = _env.reset()

    if _env._current_patient:
        _current_ground_truth = _env._current_patient["ground_truth"]

    return {
        "task": task,
        "difficulty": diff,
        "episode_id": result["episode_id"],
        "observation": result["observation"],
        "step": 0,
        "max_steps": result["max_steps"],
        "done": False,
    }

@app.post("/step")
async def step(req: StepRequest):
    """Submit agent action. Must call /reset first."""
    global _episode_reward_sum, _episode_steps, _last_grader_score
    global _current_ground_truth, _episode_done

    if _env is None:
        raise HTTPException(400, "Call /reset before /step")

    risk_score = _clamp(float(req.risk_score))
    confidence = _clamp(float(req.confidence))
    urgency = str(req.urgency_level).lower().strip()
    if urgency not in {"low", "medium", "high", "critical"}:
        urgency = "medium"

    action = AgentAction(
        patient_id=str(req.patient_id),
        risk_score=risk_score,
        recommended_intervention=str(req.recommended_intervention),
        urgency_level=urgency,
        reasoning=str(req.reasoning),
        confidence=confidence,
    )

    result = _env.step(action)

    ground_truth = _current_ground_truth or {
        "true_risk": result["reward_breakdown"].get("true_risk", risk_score),
        "ideal_intervention": result["reward_breakdown"].get("ideal_intervention", req.recommended_intervention),
    }

    if _env._current_patient:
        _current_ground_truth = _env._current_patient["ground_truth"]

    grader_score = _grade(_task, action.model_dump(), ground_truth)
    _last_grader_score = grader_score

    reward = _clamp(float(result["reward"]))
    _episode_reward_sum += reward
    _episode_steps += 1
    _episode_done = result["done"]

    return {
        "observation": result.get("next_observation", result["observation"]),
        "reward": reward,
        "grader_score": grader_score,
        "done": result["done"],
        "step": result["step"],
        "reward_breakdown": result["reward_breakdown"],
        "info": result.get("info", {}),
    }

@app.get("/state")
async def get_state():
    if _env is None:
        return {"status": "not_started"}
    return {
        "task": _task,
        "difficulty": _difficulty,
        "steps": _episode_steps,
        "episode_done": _episode_done,
        "mean_reward": round(_episode_reward_sum / max(1, _episode_steps), 4),
        "last_grader_score": _last_grader_score,
    }

# ── Hackathon endpoints ───────────────────────────────────────────────────────
TASKS = [
    {
        "task_id": "triage",
        "name": "Patient Triage",
        "difficulty": "easy",
        "max_steps": 5,
        "description": (
            "Prioritize patients by urgency based on clear behavioral signal patterns. "
            "Match urgency_level (low/medium/high/critical) to true patient risk."
        ),
        "real_world_context": "1 in 5 adults worldwide experience a mental health condition",
    },
    {
        "task_id": "risk_stratification",
        "name": "Risk Stratification",
        "difficulty": "medium",
        "max_steps": 5,
        "description": (
            "Classify patients into risk tiers using noisy, partially conflicting signals. "
            "Requires nuanced interpretation of behavioral drift patterns."
        ),
        "real_world_context": "Early detection reduces mental health crisis hospitalizations by 40%",
    },
    {
        "task_id": "early_warning",
        "name": "Early Warning Detection",
        "difficulty": "hard",
        "max_steps": 5,
        "description": (
            "Detect early deterioration before it becomes critical. High noise, "
            "limited history. False negatives (missing high-risk patients) are severely penalized."
        ),
        "real_world_context": "Passive sensing detects depression onset 2-3 weeks before clinical presentation",
    },
]

ACTION_SCHEMA = {
    "risk_score": {"type": "float", "range": "0.06-0.94", "description": "Predicted patient risk score"},
    "recommended_intervention": {
        "type": "categorical",
        "options": ["no_action", "psychoeducation_materials", "peer_support_referral",
                    "therapist_consultation", "medication_review", "wellness_check",
                    "crisis_hotline_referral", "emergency_services"],
    },
    "urgency_level": {"type": "categorical", "options": ["low", "medium", "high", "critical"]},
    "reasoning": {"type": "string", "description": "Agent's reasoning for assessment"},
    "confidence": {"type": "float", "range": "0.06-0.94"},
}

SIGNAL_GUIDE = {
    "sleep_hours": "Normal: 7-9h. <5h or >10h = risk signal.",
    "activity_steps": "Normal: 5000-10000. <2000 = sedentary depression risk.",
    "screen_time_hours": ">8h/day = anxiety/mood disorder correlation.",
    "social_interactions": "<2/day = social withdrawal (key depression indicator).",
    "heart_rate_variability": "Low HRV (<20ms) = autonomic stress.",
    "app_usage_variance": "High variance (>0.6) = possible manic episode.",
    "typing_speed_change": "< -0.15 = psychomotor retardation.",
    "location_entropy": "<0.3 = social isolation.",
}

@app.get("/tasks")
async def get_tasks():
    return {
        "tasks": TASKS,
        "action_schema": ACTION_SCHEMA,
        "signal_guide": SIGNAL_GUIDE,
        "total_tasks": len(TASKS),
        "impact": "Passive monitoring can reach 1 billion unserved mental health patients",
    }

@app.get("/grader")
async def get_grader():
    """Score strictly between 0 and 1. Blend of mean reward and last grader score."""
    if _env is None or _episode_steps == 0:
        score = _clamp(0.5)
        return {
            "score": score,
            "task": _task,
            "difficulty": _difficulty,
            "steps_completed": 0,
            "episode_done": False,
            "mean_reward": score,
            "last_grader_score": score,
            "note": "No steps yet — returning baseline score",
        }

    mean_reward = _clamp(_episode_reward_sum / max(1, _episode_steps))
    score = _clamp(0.6 * mean_reward + 0.4 * _last_grader_score)

    return {
        "score": score,
        "task": _task,
        "difficulty": _difficulty,
        "steps_completed": _episode_steps,
        "episode_done": _episode_done,
        "mean_reward": round(mean_reward, 4),
        "last_grader_score": round(_last_grader_score, 4),
    }

@app.get("/baseline")
async def get_baseline():
    """Run median baseline agent on all 3 tasks."""
    results = {}
    for task_def in TASKS:
        tid = task_def["task_id"]
        score = _run_baseline_episode(tid)
        results[tid] = {"score": score, "difficulty": task_def["difficulty"]}
    return {
        "baseline_scores": results,
        "policy": "Median baseline — always predicts risk_score=0.5, urgency=medium",
        "note": "LLM agent should significantly outperform this baseline",
    }

def _run_baseline_episode(task: str) -> float:
    env = MoodMapEnv(task=task, difficulty="easy", max_steps=5)
    env.reset()
    total = 0.0
    steps = 0
    for _ in range(5):
        if env._current_patient is None:
            break
        gt = env._current_patient["ground_truth"]
        action = AgentAction(
            patient_id="baseline",
            risk_score=0.5,
            recommended_intervention="therapist_consultation",
            urgency_level="medium",
            reasoning="median baseline",
            confidence=0.5,
        )
        result = env.step(action)
        total += _grade(task, action.model_dump(), gt)
        steps += 1
        if result["done"]:
            break
    return round(_clamp(total / max(1, steps)), 4)

# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    try:
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>MoodMap API running — <a href='/docs'>API Docs</a></h1>")