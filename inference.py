"""
MoodMap Inference Script — OpenEnv Compatible
Runs an LLM agent through the MoodMap environment using the OpenAI client.

Required env vars:
  API_BASE_URL  — e.g. https://api-inference.huggingface.co/v1
  MODEL_NAME    — e.g. Qwen/Qwen2.5-72B-Instruct
  HF_TOKEN      — your Hugging Face token

Logs follow the required hackathon format:
  [START] task=<task> env=moodmap model=<model>
  [STEP] step=<n> action=<json> reward=<r> done=<bool> error=<null|msg>
  [END] success=<bool> steps=<n> score=<score> rewards=<r1,r2,...>
"""
import os
import sys
import json
import time

from openai import OpenAI
from moodmap_env import MoodMapEnv
from moodmap_env.models import AgentAction
from graders import grade

# ── Environment variables ─────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("[WARN] HF_TOKEN not set. Using fallback actions.", flush=True)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

ENV_NAME = "moodmap"

# ── Required hackathon log format ─────────────────────────────
def log_start(task, model):
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    action_str = json.dumps(action) if isinstance(action, dict) else str(action)
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical AI assistant specialized in passive mental health monitoring.

Analyze behavioral signals from wearables and smartphones to assess patient mental health risk.

For each patient, respond with ONLY valid JSON in this exact format:
{
  "risk_score": 0.45,
  "recommended_intervention": "peer_support_referral",
  "urgency_level": "medium",
  "reasoning": "Brief reasoning here",
  "confidence": 0.78
}

risk_score and confidence must be strictly between 0.05 and 0.95.
urgency_level: low, medium, high, or critical
recommended_intervention: no_action, psychoeducation_materials, peer_support_referral,
  therapist_consultation, medication_review, wellness_check, crisis_hotline_referral, emergency_services"""


def build_prompt(obs: dict) -> str:
    signals = obs.get("signals", {})
    med = obs.get("medication_adherence")
    med_str = f"{med:.0%}" if med else "not applicable"
    return (
        f"Patient ID: {obs['patient_id']}\n"
        f"Age: {obs['age']} | Gender: {obs['gender']}\n"
        f"Task: {obs['task']} | Difficulty: {obs['difficulty']}\n"
        f"History: {obs['history_days']} days | Prior episodes: {obs['prior_episodes']}\n"
        f"Baseline mood: {obs['baseline_mood_score']:.2f} | Medication adherence: {med_str}\n\n"
        f"Behavioral Signals (last 7 days):\n"
        f"  Sleep: {signals.get('sleep_hours', 0):.1f} hrs/night\n"
        f"  Activity: {signals.get('activity_steps', 0):,} steps/day\n"
        f"  Screen time: {signals.get('screen_time_hours', 0):.1f} hrs/day\n"
        f"  Social interactions: {signals.get('social_interactions', 0)}/week\n"
        f"  HRV: {signals.get('heart_rate_variability', 0):.1f} ms\n"
        f"  App usage variance: {signals.get('app_usage_variance', 0):.3f}\n"
        f"  Typing speed change: {signals.get('typing_speed_change', 0):+.3f}\n"
        f"  Location entropy: {signals.get('location_entropy', 0):.3f}\n\n"
        f"Assess this patient's mental health risk and recommend an appropriate intervention."
    )


def call_llm(prompt: str, max_retries: int = 3) -> dict:
    if not HF_TOKEN:
        return {
            "risk_score": 0.50,
            "recommended_intervention": "wellness_check",
            "urgency_level": "medium",
            "reasoning": "HF_TOKEN not set. Using safe default.",
            "confidence": 0.10,
        }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            action = json.loads(raw)
            action["risk_score"] = max(0.05, min(0.95, float(action.get("risk_score", 0.5))))
            action["confidence"] = max(0.05, min(0.95, float(action.get("confidence", 0.5))))
            return action
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "risk_score": 0.50,
                    "recommended_intervention": "wellness_check",
                    "urgency_level": "medium",
                    "reasoning": f"LLM failed after {max_retries} attempts.",
                    "confidence": 0.10,
                }


def run_episode(task: str = "triage", difficulty: str = "easy", max_steps: int = 5) -> dict:
    env = MoodMapEnv(task=task, difficulty=difficulty, max_steps=max_steps)

    log_start(task, MODEL_NAME)

    state = env.reset()
    obs = state["observation"]

    step_num = 0
    rewards = []
    done = False

    while not done:
        step_num += 1
        prompt = build_prompt(obs)

        error = None
        try:
            action_dict = call_llm(prompt)
            action_dict["patient_id"] = obs["patient_id"]
            action = AgentAction(**action_dict)
            result = env.step(action)

            reward = result["reward"]
            done = result["done"]
            rewards.append(reward)

            log_action = {
                "risk_score": action_dict["risk_score"],
                "recommended_intervention": action_dict["recommended_intervention"],
                "urgency_level": action_dict["urgency_level"],
            }
            log_step(step_num, log_action, reward, done)

            if not done:
                obs = result.get("next_observation", obs)

        except Exception as e:
            error = str(e)
            log_step(step_num, {}, 0.0, True, error=error)
            break

    summary = env.get_episode_summary()
    total_reward = summary.get("total_reward", sum(rewards))
    mean_reward  = summary.get("mean_reward", total_reward / max(step_num, 1))

    score = max(0.0001, min(0.9999, mean_reward))
    success = score > 0.5

    log_end(success, step_num, score, rewards)
    return summary


def main():
    tasks = [
        ("triage",              "easy",   5),
        ("risk_stratification", "medium", 5),
        ("early_warning",       "hard",   5),
    ]

    scores = []
    for task, difficulty, max_steps in tasks:
        result = run_episode(task=task, difficulty=difficulty, max_steps=max_steps)
        mean = result.get("mean_reward", 0.0)
        scores.append(max(0.0001, min(0.9999, mean)))
        time.sleep(0.5)

    avg = sum(scores) / len(scores)
    print(f"\n[SUMMARY] average_score={avg:.4f} scores={','.join(f'{s:.4f}' for s in scores)}", flush=True)
    print(f"[IMPACT]  MoodMap agent assessed mental health risk across all 3 task scenarios", flush=True)


if __name__ == "__main__":
    main()