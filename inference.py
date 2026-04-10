"""
MoodMap Inference Script — OpenEnv Compatible
Runs an LLM agent through the MoodMap environment using the OpenAI client.

Required env vars:
  API_BASE_URL  — e.g. https://api-inference.huggingface.co/v1
  MODEL_NAME    — e.g. Qwen/Qwen2.5-72B-Instruct
  HF_TOKEN      — your Hugging Face token (NO DEFAULT)
  
Optional:
  LOCAL_IMAGE_NAME — if using from_docker_image()
"""
import os
import sys
import json
import time

from openai import OpenAI
from moodmap_env import MoodMapEnv
from graders import grade

# ── Environment variables (with sensible defaults for API_BASE_URL and MODEL_NAME) ──────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")          # NO default — must be set in environment

# Check if HF_TOKEN is set (but don't fail hard - allow demo mode)
if not HF_TOKEN:
    print("[WARN] HF_TOKEN not set. LLM calls will fail. Set HF_TOKEN as a secret in Hugging Face Spaces.", flush=True)

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional

# All LLM calls use the OpenAI client configured via these variables
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are a clinical AI assistant specialized in passive mental health monitoring.

You analyze behavioral signals from wearables and smartphones to assess patient mental health risk.

For each patient observation you receive, you must:
1. Estimate a risk score between 0.05 and 0.95 (never exactly 0 or 1)
2. Recommend an appropriate intervention
3. Assess urgency level
4. Provide clear reasoning

Available interventions:
- no_action
- psychoeducation_materials
- peer_support_referral
- therapist_consultation
- crisis_hotline_referral
- emergency_services
- medication_review
- wellness_check

Urgency levels: low, medium, high, critical

CRITICAL: Always respond with ONLY valid JSON in this exact format:
{
  "risk_score": 0.45,
  "recommended_intervention": "peer_support_referral",
  "urgency_level": "medium",
  "reasoning": "Patient shows declining sleep and reduced social interaction over 14 days...",
  "confidence": 0.78
}

risk_score and confidence must be strictly between 0.05 and 0.95."""


def build_prompt(obs: dict) -> str:
    signals = obs.get("signals", {})
    med = obs.get("medication_adherence")
    med_str = f"{med:.0%}" if med else "not applicable"
    return f"""Patient ID: {obs['patient_id']}
Age: {obs['age']} | Gender: {obs['gender']}
Task: {obs['task']} | Difficulty: {obs['difficulty']}
History: {obs['history_days']} days | Prior episodes: {obs['prior_episodes']}
Baseline mood: {obs['baseline_mood_score']:.2f} | Medication adherence: {med_str}

Behavioral Signals (last 7 days):
  Sleep: {signals.get('sleep_hours', '?'):.1f} hrs/night
  Activity: {signals.get('activity_steps', '?'):,} steps/day
  Screen time: {signals.get('screen_time_hours', '?'):.1f} hrs/day
  Social interactions: {signals.get('social_interactions', '?')}/week
  HRV: {signals.get('heart_rate_variability', '?'):.1f} ms
  App usage variance: {signals.get('app_usage_variance', '?'):.3f}
  Typing speed change: {signals.get('typing_speed_change', '?'):+.3f}
  Location entropy: {signals.get('location_entropy', '?'):.3f}

Assess this patient's mental health risk and recommend an appropriate intervention."""


def call_llm(prompt: str, max_retries: int = 3) -> dict:
    # If HF_TOKEN not set, return a fallback action immediately
    if not HF_TOKEN:
        return {
            "risk_score": 0.50,
            "recommended_intervention": "wellness_check",
            "urgency_level": "medium",
            "reasoning": "HF_TOKEN not set. Using safe default action.",
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
            # Strip any markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()
            action = json.loads(raw)

            # Validate and clamp risk_score & confidence
            action["risk_score"] = max(0.05, min(0.95, float(action.get("risk_score", 0.5))))
            action["confidence"] = max(0.05, min(0.95, float(action.get("confidence", 0.5))))
            return action
        except Exception as e:
            print(f"[WARN] LLM call attempt {attempt+1} failed: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                # Fallback action
                return {
                    "risk_score": 0.50,
                    "recommended_intervention": "wellness_check",
                    "urgency_level": "medium",
                    "reasoning": f"LLM call failed after {max_retries} attempts. Using safe default.",
                    "confidence": 0.10,
                }


def run_episode(task: str = "triage", difficulty: str = "easy", max_steps: int = 5):
    env = MoodMapEnv(task=task, difficulty=difficulty, max_steps=max_steps)

    # ── [START] ──────────────────────────────────────────────────────────────
    print("[START]", flush=True)
    print(json.dumps({"task": task, "difficulty": difficulty, "max_steps": max_steps}), flush=True)

    state = env.reset()
    episode_id = state["episode_id"]
    total_reward = 0.0
    step_num = 0

    obs = state["observation"]

    while True:
        step_num += 1
        prompt = build_prompt(obs)
        action_dict = call_llm(prompt)

        # Build action with patient_id
        action_dict["patient_id"] = obs["patient_id"]

        # ── [STEP] ─────────────────────────────────────────────────────────
        print("[STEP]", flush=True)
        print(json.dumps({
            "step": step_num,
            "patient_id": obs["patient_id"],
            "task": task,
            "difficulty": difficulty,
            "action": {
                "risk_score": action_dict["risk_score"],
                "recommended_intervention": action_dict["recommended_intervention"],
                "urgency_level": action_dict["urgency_level"],
                "confidence": action_dict["confidence"],
            }
        }), flush=True)

        from moodmap_env.models import AgentAction
        action = AgentAction(**action_dict)
        result = env.step(action)

        reward = result["reward"]
        total_reward += reward

        print(json.dumps({
            "step": step_num,
            "reward": reward,
            "breakdown": result["reward_breakdown"],
            "done": result["done"],
        }), flush=True)

        if result["done"]:
            obs = None
            break
        else:
            obs = result.get("next_observation", obs)

    summary = env.get_episode_summary()

    # ── [END] ─────────────────────────────────────────────────────────────
    print("[END]", flush=True)
    print(json.dumps({
        "episode_id": episode_id,
        "task": task,
        "difficulty": difficulty,
        "total_reward": summary.get("total_reward", round(total_reward, 4)),
        "mean_reward": summary.get("mean_reward", round(total_reward / max(step_num, 1), 4)),
        "steps": step_num,
    }), flush=True)

    return summary


def run_all_tasks():
    """Run through all task/difficulty combinations."""
    configs = [
        ("triage", "easy"),
        ("triage", "medium"),
        ("risk_stratification", "easy"),
        ("risk_stratification", "medium"),
        ("risk_stratification", "hard"),
        ("early_warning", "medium"),
        ("early_warning", "hard"),
    ]

    all_results = []
    for task, difficulty in configs:
        print(f"\n{'='*60}", flush=True)
        print(f"Running: {task} / {difficulty}", flush=True)
        print('='*60, flush=True)
        result = run_episode(task=task, difficulty=difficulty, max_steps=5)
        all_results.append(result)
        time.sleep(0.5)

    print("\n\n" + "="*60, flush=True)
    print("ALL EPISODES COMPLETE", flush=True)
    print("="*60, flush=True)
    for r in all_results:
        print(f"  {r.get('task','?'):25s} {r.get('difficulty','?'):8s}  mean={r.get('mean_reward','?'):.4f}  total={r.get('total_reward','?'):.4f}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MoodMap inference runner")
    parser.add_argument("--task", default="triage", choices=["triage", "risk_stratification", "early_warning"])
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="Run all task/difficulty combos")
    args = parser.parse_args()

    if args.all:
        run_all_tasks()
    else:
        run_episode(task=args.task, difficulty=args.difficulty, max_steps=args.steps)
    #done