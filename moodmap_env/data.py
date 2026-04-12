import random
import uuid
from datetime import datetime, timedelta
from .models import PatientObservation, BehavioralSignals


DIFFICULTY_CONFIG = {
    "easy": {
        "signal_noise": 0.05,
        "history_days_range": (60, 90),
        "ambiguity": 0.1,
        "description": "Clear behavioral patterns with minimal noise",
    },
    "medium": {
        "signal_noise": 0.20,
        "history_days_range": (30, 60),
        "ambiguity": 0.35,
        "description": "Moderate noise with some conflicting signals",
    },
    "hard": {
        "signal_noise": 0.45,
        "history_days_range": (7, 30),
        "ambiguity": 0.65,
        "description": "High noise, conflicting signals, limited history",
    },
}

TASKS = ["triage", "risk_stratification", "early_warning"]

INTERVENTIONS = [
    "no_action",
    "psychoeducation_materials",
    "peer_support_referral",
    "therapist_consultation",
    "crisis_hotline_referral",
    "emergency_services",
    "medication_review",
    "wellness_check",
]

GENDERS = ["male", "female", "non-binary"]


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _noisy(val, noise, lo=0.0, hi=1.0):
    return _clamp(val + random.gauss(0, noise), lo, hi)


def generate_patient(
    task: str = None,
    difficulty: str = None,
    true_risk: float = None,
    seed: int = None,
) -> dict:
    """
    Returns a (PatientObservation, ground_truth) tuple dict.
    ground_truth contains the hidden true_risk and ideal_intervention.
    Pass seed for reproducible patient generation.
    """
    if seed is not None:
        random.seed(seed)
    task = task or random.choice(TASKS)
    difficulty = difficulty or random.choice(list(DIFFICULTY_CONFIG.keys()))
    cfg = DIFFICULTY_CONFIG[difficulty]
    noise = cfg["signal_noise"]

    # True risk in (0.05, 0.95) — never extreme
    if true_risk is None:
        true_risk = _clamp(random.betavariate(2, 2), 0.08, 0.92)

    # Derive signals from true risk + noise
    age = random.randint(18, 65)
    gender = random.choice(GENDERS)
    history_days = random.randint(*cfg["history_days_range"])
    prior_episodes = int(true_risk * 4 + random.randint(0, 2))

    # Higher risk → worse sleep, less activity, more screen time, less social
    sleep_hours = _noisy(8.0 - true_risk * 4.0, noise * 3, 2, 12)
    activity_steps = int(_noisy(8000 - true_risk * 6000, noise * 3000, 200, 15000))
    screen_time_hours = _noisy(2.0 + true_risk * 6.0, noise * 2, 0.5, 12)
    social_interactions = int(_noisy(6 - true_risk * 5, noise * 3, 0, 15))
    hrv = _noisy(55 - true_risk * 30, noise * 15, 10, 100)
    app_variance = _noisy(true_risk * 0.8, noise * 0.3, 0, 1)
    typing_speed_change = _noisy(-true_risk * 0.6, noise * 0.4, -1, 1)
    location_entropy = _noisy(1.0 - true_risk * 0.7, noise * 0.3, 0, 1)

    # Medication adherence only for some patients
    medication_adherence = None
    if random.random() < 0.6:
        medication_adherence = _noisy(1.0 - true_risk * 0.5, noise * 0.3, 0, 1)

    signals = BehavioralSignals(
        sleep_hours=round(sleep_hours, 2),
        activity_steps=activity_steps,
        screen_time_hours=round(screen_time_hours, 2),
        social_interactions=social_interactions,
        heart_rate_variability=round(hrv, 2),
        app_usage_variance=round(app_variance, 3),
        typing_speed_change=round(typing_speed_change, 3),
        location_entropy=round(location_entropy, 3),
    )

    baseline_mood = _noisy(0.7 - true_risk * 0.4, noise * 0.15, 0.1, 0.9)

    obs = PatientObservation(
        patient_id=f"PT-{uuid.uuid4().hex[:8].upper()}",
        timestamp=(datetime.utcnow() - timedelta(minutes=random.randint(0, 120))).isoformat(),
        age=age,
        gender=gender,
        baseline_mood_score=round(baseline_mood, 3),
        signals=signals,
        history_days=history_days,
        prior_episodes=prior_episodes,
        medication_adherence=round(medication_adherence, 3) if medication_adherence else None,
        task=task,
        difficulty=difficulty,
    )

    # Ideal intervention based on true risk
    if true_risk < 0.25:
        ideal_intervention = "no_action"
    elif true_risk < 0.45:
        ideal_intervention = "psychoeducation_materials"
    elif true_risk < 0.60:
        ideal_intervention = "peer_support_referral"
    elif true_risk < 0.75:
        ideal_intervention = "therapist_consultation"
    elif true_risk < 0.88:
        ideal_intervention = "crisis_hotline_referral"
    else:
        ideal_intervention = "emergency_services"

    return {
        "observation": obs,
        "ground_truth": {
            "true_risk": round(true_risk, 4),
            "ideal_intervention": ideal_intervention,
            "difficulty": difficulty,
            "task": task,
        },
    }


def generate_batch(n: int = 5, task: str = None, difficulty: str = None) -> list:
    return [generate_patient(task=task, difficulty=difficulty) for _ in range(n)]