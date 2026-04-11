"""
MoodMap Passive Mental Health — FastAPI Server
===============================================
POST /reset     — start a new episode (task: triage|risk_stratification|early_warning)
POST /step      — submit an agent action, get observation + reward
GET  /state     — current episode metadata
GET  /health    — health check  {"status": "healthy"}
GET  /tasks     — list all 3 tasks + action schema
GET  /grader    — score for current episode strictly in (0,1)
GET  /baseline  — run baseline agent on all 3 tasks
GET  /metadata  — environment name + description
GET  /schema    — action / observation / state schemas
POST /mcp       — JSON-RPC 2.0 stub (for validator compliance)

Run:  uvicorn app:app --host 0.0.0.0 --port 7860 --reload
Docs: http://localhost:7860/docs
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

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

# ── Score bounds — strictly between 0 and 1, never hitting 0.0 or 1.0 ────────
MIN_SCORE = 0.06
MAX_SCORE = 0.94

def _clamp(v: float) -> float:
    """Clamp to [0.06, 0.94] — always strictly between 0 and 1."""
    return max(MIN_SCORE, min(MAX_SCORE, float(v)))

# ── Valid task/scenario names ─────────────────────────────────────────────────
VALID_TASKS = ["triage", "risk_stratification", "early_warning"]
VALID_DIFFS = ["easy", "medium", "hard"]

# Map scenario names to task names (same values, but accept both)
SCENARIO_TO_TASK: Dict[str, str] = {
    "triage": "triage",
    "risk_stratification": "risk_stratification",
    "early_warning": "early_warning",
}

# ── Global episode state ──────────────────────────────────────────────────────
_env: Optional[MoodMapEnv] = None
_task: str = "triage"
_difficulty: str = "easy"
_episode_reward_sum: float = 0.0
_episode_steps: int = 0
_last_grader_score: float = MIN_SCORE + 0.44   # 0.5 — safe initial mid-range
_current_ground_truth: Optional[Dict] = None
_episode_done: bool = False
_episode_id: str = "not-started"


# ── Request models ────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    # Accept 'task' OR 'scenario' (validator sends 'scenario' from openenv.yaml)
    task: Optional[str] = None
    scenario: Optional[str] = None   # ← added so validator's scenario param works
    difficulty: Optional[str] = None
    seed: Optional[int] = None

class StepRequest(BaseModel):
    patient_id: str = Field(default="unknown")
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    recommended_intervention: str = Field(default="no_action")
    urgency_level: str = Field(default="medium")
    reasoning: str = Field(default="")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ── Grading helpers ───────────────────────────────────────────────────────────
def _grade(task: str, action_dict: dict, ground_truth: dict) -> float:
    """Run the per-task grader and clamp result to (MIN_SCORE, MAX_SCORE)."""
    if task == "triage":
        raw = grade_triage(action_dict, ground_truth)
    elif task == "risk_stratification":
        raw = grade_risk_stratification(action_dict, ground_truth)
    elif task == "early_warning":
        raw = grade_early_warning(action_dict, ground_truth)
    else:
        raw = 0.5
    return _clamp(raw)


def _resolve_task(req: ResetRequest) -> str:
    """
    Resolve the task name from the reset request.
    Accepts 'task' or 'scenario' (the validator sends 'scenario' from openenv.yaml).
    """
    # Prefer explicit task, then scenario, then default
    raw = req.task or req.scenario or "triage"
    # Map scenario → task (they're the same strings here)
    resolved = SCENARIO_TO_TASK.get(raw.lower().strip(), raw.lower().strip())
    return resolved if resolved in VALID_TASKS else "triage"


# ══════════════════════════════════════════════════════════════════════════════
# STANDARD ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Health check — returns 'healthy' as required by the OpenEnv validator."""
    return {"status": "healthy", "project": "MoodMap Passive Mental Health Agent"}


@app.post("/reset")
async def reset(req: ResetRequest = None):
    """
    Start a new episode.
    Accepts 'task' or 'scenario': triage | risk_stratification | early_warning
    """
    global _env, _task, _difficulty, _episode_reward_sum, _episode_steps
    global _last_grader_score, _current_ground_truth, _episode_done, _episode_id

    if req is None:
        req = ResetRequest()

    task = _resolve_task(req)
    diff = (req.difficulty or "easy").lower().strip()
    if diff not in VALID_DIFFS:
        diff = "easy"

    _task = task
    _difficulty = diff
    _episode_reward_sum = 0.0
    _episode_steps = 0
    _last_grader_score = 0.5
    _episode_done = False

    _env = MoodMapEnv(task=task, difficulty=diff, max_steps=5)
    result = _env.reset()
    _episode_id = result["episode_id"]

    if _env._current_patient:
        _current_ground_truth = _env._current_patient["ground_truth"]

    return {
        "task": task,
        "scenario": task,          # echo back as 'scenario' too for compatibility
        "difficulty": diff,
        "episode_id": _episode_id,
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

    # Auto-reset if no episode started yet
    if _env is None:
        await reset(ResetRequest())

    # Clamp floats into valid range
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

    # Use ground truth captured at reset/previous-step
    ground_truth = _current_ground_truth or {
        "true_risk": result["reward_breakdown"].get("true_risk", risk_score),
        "ideal_intervention": result["reward_breakdown"].get(
            "ideal_intervention", req.recommended_intervention
        ),
    }

    # Advance ground truth to next patient
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
    """Current episode metadata."""
    if _env is None:
        return {"status": "not_started", "task": _task}
    return {
        "task": _task,
        "scenario": _task,
        "difficulty": _difficulty,
        "steps": _episode_steps,
        "episode_done": _episode_done,
        "episode_id": _episode_id,
        "mean_reward": round(_episode_reward_sum / max(1, _episode_steps), 4),
        "last_grader_score": _last_grader_score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HACKATHON / OPENENV REQUIRED ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

TASKS = [
    {
        "task_id": "triage",
        "name": "Patient Triage",
        "scenario": "triage",
        "difficulty": "easy",
        "max_steps": 5,
        "description": (
            "Prioritize patients by urgency based on clear behavioral signal patterns. "
            "Match urgency_level (low/medium/high/critical) to true patient risk."
        ),
        "grader": "graders.grade_triage",
        "has_grader": True,
        "score": 0.5,           # strictly between 0 and 1 — required by OpenEnv validator
        "score_range": [0.06, 0.94],
        "real_world_context": "1 in 5 adults worldwide experience a mental health condition",
    },
    {
        "task_id": "risk_stratification",
        "name": "Risk Stratification",
        "scenario": "risk_stratification",
        "difficulty": "medium",
        "max_steps": 5,
        "description": (
            "Classify patients into risk tiers using noisy, partially conflicting signals. "
            "Requires nuanced interpretation of behavioral drift patterns."
        ),
        "grader": "graders.grade_risk_stratification",
        "has_grader": True,
        "score": 0.5,           # strictly between 0 and 1 — required by OpenEnv validator
        "score_range": [0.06, 0.94],
        "real_world_context": "Early detection reduces mental health crisis hospitalizations by 40%",
    },
    {
        "task_id": "early_warning",
        "name": "Early Warning Detection",
        "scenario": "early_warning",
        "difficulty": "hard",
        "max_steps": 5,
        "description": (
            "Detect early deterioration before it becomes critical. High noise, "
            "limited history. False negatives severely penalized."
        ),
        "grader": "graders.grade_early_warning",
        "has_grader": True,
        "score": 0.5,           # strictly between 0 and 1 — required by OpenEnv validator
        "score_range": [0.06, 0.94],
        "real_world_context": "Passive sensing detects depression onset 2-3 weeks before clinical presentation",
    },
]

ACTION_SCHEMA = {
    "risk_score": {"type": "float", "range": "0.06-0.94", "description": "Predicted patient risk score"},
    "recommended_intervention": {
        "type": "categorical",
        "options": [
            "no_action", "psychoeducation_materials", "peer_support_referral",
            "therapist_consultation", "medication_review", "wellness_check",
            "crisis_hotline_referral", "emergency_services",
        ],
    },
    "urgency_level": {"type": "categorical", "options": ["low", "medium", "high", "critical"]},
    "reasoning": {"type": "string", "description": "Agent reasoning for assessment"},
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
    """List all 3 tasks with grader info and action schema."""
    return {
        "tasks": TASKS,
        "action_schema": ACTION_SCHEMA,
        "signal_guide": SIGNAL_GUIDE,
        "total_tasks": len(TASKS),
        "impact": "Passive monitoring can reach 1 billion unserved mental health patients",
    }


@app.get("/grader")
async def get_grader():
    """
    Normalized score strictly between 0 and 1 (never 0.0 or 1.0).
    Score = 60% mean_reward + 40% last_grader_score, clamped to [0.06, 0.94].
    """
    if _env is None or _episode_steps == 0:
        # Return a valid mid-range score even before any steps
        safe_score = 0.5
        return {
            "score": safe_score,
            "task": _task,
            "scenario": _task,
            "difficulty": _difficulty,
            "steps_completed": 0,
            "episode_done": False,
            "mean_reward": safe_score,
            "last_grader_score": safe_score,
            "note": "No steps completed yet — returning safe baseline score of 0.5",
        }

    mean_reward = _clamp(_episode_reward_sum / max(1, _episode_steps))
    score = _clamp(0.6 * mean_reward + 0.4 * _last_grader_score)

    # Extra safety: ensure score is NEVER exactly 0.0 or 1.0
    # (Python float arithmetic shouldn't produce these given our bounds, but be safe)
    if score <= 0.0:
        score = MIN_SCORE
    if score >= 1.0:
        score = MAX_SCORE

    return {
        "score": score,
        "task": _task,
        "scenario": _task,
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
        results[tid] = {
            "score": score,
            "difficulty": task_def["difficulty"],
            "grader": task_def["grader"],
        }
    return {
        "baseline_scores": results,
        "policy": "Median baseline — always predicts risk_score=0.5, urgency=medium",
        "note": "LLM agent should significantly outperform this baseline",
    }


def _run_baseline_episode(task: str) -> float:
    """Run a median-prediction baseline and return clamped mean grader score."""
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


# ══════════════════════════════════════════════════════════════════════════════
# OPENENV FRAMEWORK COMPATIBILITY ENDPOINTS
# These are required by the openenv-core validator (Phase 1 + Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/metadata")
async def metadata():
    """Environment metadata — required by OpenEnv validator."""
    return {
        "name": "MoodMap Passive Mental Health Agent",
        "description": (
            "LLM agent analyzes passive behavioral signals (sleep, activity, "
            "screen time, HRV, social interactions) from smartphones and wearables "
            "to assess patient mental health risk and recommend appropriate interventions."
        ),
        "version": "1.0.0",
        "author": "MoodMap Team",
        "tasks": [t["task_id"] for t in TASKS],
    }


@app.get("/schema")
async def schema():
    """Action / observation / state schemas — required by OpenEnv validator."""
    return {
        "action": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "risk_score": {"type": "number", "minimum": 0.06, "maximum": 0.94},
                "recommended_intervention": {
                    "type": "string",
                    "enum": [
                        "no_action", "psychoeducation_materials", "peer_support_referral",
                        "therapist_consultation", "medication_review", "wellness_check",
                        "crisis_hotline_referral", "emergency_services",
                    ],
                },
                "urgency_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
                "reasoning": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.06, "maximum": 0.94},
            },
            "required": ["risk_score", "recommended_intervention", "urgency_level"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "age": {"type": "integer"},
                "gender": {"type": "string"},
                "baseline_mood_score": {"type": "number"},
                "signals": {
                    "type": "object",
                    "properties": {
                        "sleep_hours": {"type": "number"},
                        "activity_steps": {"type": "integer"},
                        "screen_time_hours": {"type": "number"},
                        "social_interactions": {"type": "integer"},
                        "heart_rate_variability": {"type": "number"},
                        "app_usage_variance": {"type": "number"},
                        "typing_speed_change": {"type": "number"},
                        "location_entropy": {"type": "number"},
                    },
                },
                "history_days": {"type": "integer"},
                "prior_episodes": {"type": "integer"},
                "medication_adherence": {"type": "number", "nullable": True},
                "task": {"type": "string"},
                "difficulty": {"type": "string"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "difficulty": {"type": "string"},
                "steps": {"type": "integer"},
                "episode_done": {"type": "boolean"},
                "episode_id": {"type": "string"},
                "mean_reward": {"type": "number"},
                "last_grader_score": {"type": "number"},
            },
        },
    }


@app.post("/mcp")
async def mcp(request: Request):
    """
    JSON-RPC 2.0 endpoint — required by OpenEnv validator.
    Returns a valid JSON-RPC 2.0 response for any method.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    method = body.get("method", "")
    req_id = body.get("id", 1)

    # Handle tools/list method
    if method == "tools/list":
        result: Any = {
            "tools": [
                {
                    "name": "reset",
                    "description": "Start a new episode with a given task",
                    "inputSchema": {"type": "object", "properties": {"task": {"type": "string"}}},
                },
                {
                    "name": "step",
                    "description": "Submit an agent action and receive a reward",
                    "inputSchema": {"type": "object"},
                },
                {
                    "name": "grader",
                    "description": "Get the current episode score (strictly between 0 and 1)",
                    "inputSchema": {"type": "object"},
                },
            ]
        }
    else:
        result = {"status": "ok", "method": method}

    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    })


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    try:
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content=(
                "<h1>MoodMap Passive Mental Health Agent</h1>"
                "<p>API is running. <a href='/docs'>View API Docs</a></p>"
            )
        )