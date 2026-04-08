from fastapi import FastAPI, Body
from typing import Optional, Dict, Any
import uuid
import random

app = FastAPI(title="MoodMap Passive Mental Health Environment")

# -------- MOCK ENV (replace with your real env if needed) -------- #

sessions = {}

def generate_observation():
    return {
        "patient_id": f"PT-{uuid.uuid4().hex[:8].upper()}",
        "timestamp": "2026-04-08T00:00:00",
        "age": random.randint(18, 70),
        "gender": random.choice(["male", "female", "non-binary"]),
        "baseline_mood_score": round(random.uniform(0.3, 0.8), 3),
        "signals": [],
        "history_days": random.randint(10, 100),
        "prior_episodes": random.randint(0, 5),
        "medication_adherence": round(random.uniform(0.3, 0.9), 3),
        "task": "triage",
        "difficulty": "easy"
    }

# -------- ROUTES -------- #

@app.get("/")
def home():
    return {"message": "MoodMap API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {"env": "MoodMap", "version": "1.0"}

# ✅ FIXED RESET (IMPORTANT)
@app.post("/reset")
def reset(body: Optional[Dict[str, Any]] = Body(default=None)):
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    episode_id = f"EP-{uuid.uuid4().hex[:10].upper()}"

    observation = generate_observation()

    sessions[session_id] = {
        "step": 0,
        "max_steps": 5,
        "episode_id": episode_id,
        "last_observation": observation
    }

    return {
        "session_id": session_id,
        "episode_id": episode_id,
        "observation": observation,
        "task": "triage",
        "difficulty": "easy",
        "step": 0,
        "max_steps": 5
    }

# ✅ STEP
@app.post("/step")
def step(body: Dict[str, Any]):
    session_id = body.get("session_id")
    patient_id = body.get("patient_id")

    if session_id not in sessions:
        return {"error": "Invalid session_id"}

    session = sessions[session_id]
    session["step"] += 1

    # fake reward logic
    reward = round(random.uniform(0.2, 0.9), 4)

    next_obs = generate_observation()

    done = session["step"] >= session["max_steps"]

    return {
        "episode_id": session["episode_id"],
        "step": session["step"],
        "observation": session["last_observation"],
        "action": body,
        "reward": reward,
        "done": done,
        "next_observation": next_obs,
        "info": {
            "task": "triage",
            "difficulty": "easy"
        },
        "grader_score": {
            "score": round(random.uniform(0.1, 1.0), 3)
        }
    }

@app.get("/sessions")
def list_sessions():
    return {"sessions": list(sessions.keys())}


# ✅ REQUIRED FOR OPENENV
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()