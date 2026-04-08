# MoodMap — Passive Mental Health OpenEnv

A reinforcement learning environment where an LLM agent analyzes **passively collected behavioral signals** (sleep, activity, screen time, HRV, social interactions) to assess patient mental health risk and recommend appropriate interventions.

## 🧠 Overview

MoodMap simulates a clinical decision support system. The agent receives observations of behavioral signals and must:
1. Estimate a **risk score** (strictly between 0.05 and 0.95)
2. Recommend an **intervention** (from a defined set)
3. Assess **urgency level** (low / medium / high / critical)
4. Provide **reasoning**

## 🎯 Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `triage` | Easy | Prioritize patients by urgency (clear signals) |
| `risk_stratification` | Medium | Classify risk tier from noisy behavioral drift |
| `early_warning` | Hard | Detect early deterioration with subtle, noisy signals |

## 📊 Reward Function

Rewards are always strictly in **(0.05, 0.95)** — never 0.0 or 1.0.

Four components:
- **Detection** (0.25–0.40): Accuracy of risk score vs true risk
- **Intervention** (0.25–0.30): Appropriateness of recommended intervention
- **Timeliness** (0.20–0.30): Urgency level match to true risk
- **Harm Avoidance** (0.10–0.20): Penalizes dangerous errors (false negatives for high-risk, over-escalation for low-risk)

## 🚀 Quick Start

### Local
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```
Open http://localhost:7860 for the live dashboard.

### Docker
```bash
docker build -t moodmap-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_token \
  -e API_BASE_URL=https://api-inference.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  moodmap-env
```

### Run Inference
```bash
export HF_TOKEN=hf_your_token
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Single task
python inference.py --task triage --difficulty easy --steps 5

# All tasks
python inference.py --all
```

## 🔌 API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Live dashboard |
| `/health` | GET | Health check |
| `/info` | GET | Environment metadata |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit action, receive reward |

### Example

```python
import requests

# Reset
r = requests.post("http://localhost:7860/reset", json={
    "task": "triage",
    "difficulty": "easy",
    "max_steps": 5
})
data = r.json()
session_id = data["session_id"]
obs = data["observation"]

# Step
result = requests.post("http://localhost:7860/step", json={
    "session_id": session_id,
    "patient_id": obs["patient_id"],
    "risk_score": 0.62,
    "recommended_intervention": "therapist_consultation",
    "urgency_level": "high",
    "reasoning": "Low sleep, reduced activity, declining HRV over 14 days.",
    "confidence": 0.75,
}).json()

print(f"Reward: {result['reward']}")  # e.g. 0.7831
```

## 📋 Pre-Submission Checklist

- ✅ Defaults set only for `API_BASE_URL` and `MODEL_NAME` (not `HF_TOKEN`)
- ✅ All LLM calls use the OpenAI client configured via these variables
- ✅ Stdout logs follow the required structured format (`[START]`/`[STEP]`/`[END]`) exactly
- ✅ Reward always in `(0.05, 0.95)` — never `0.0` or `1.0`
- ✅ Three tasks across easy/medium/hard difficulty
- ✅ Partial-credit reward with 4 components

## 🌡️ Behavioral Signals

| Signal | Description |
|--------|-------------|
| `sleep_hours` | Average nightly sleep |
| `activity_steps` | Daily step count |
| `screen_time_hours` | Daily phone screen time |
| `social_interactions` | Weekly social contacts |
| `heart_rate_variability` | HRV in milliseconds |
| `app_usage_variance` | Variance in app usage patterns |
| `typing_speed_change` | Change in typing speed (proxy for cognitive state) |
| `location_entropy` | Diversity of locations visited |

## 📁 Structure

```
moodmap/
├── moodmap_env/
│   ├── __init__.py
│   ├── models.py       # Pydantic schemas
│   ├── data.py         # Synthetic patient generator
│   └── env.py          # Core MoodMapEnv + reward function
├── graders/
│   └── __init__.py     # Task-specific graders
├── app.py              # FastAPI REST server
├── dashboard.html      # ICU-style live dashboard
├── inference.py        # LLM inference runner ([START]/[STEP]/[END])
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

## License

Apache-2.0
