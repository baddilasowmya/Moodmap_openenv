"""
Deterministic graders for MoodMap tasks.
Each grader takes an action dict and ground_truth dict and returns a score in (0.05, 0.95).
"""
import math


def _clamp(v):
    return max(0.05, min(0.95, v))


def grade_triage(action: dict, ground_truth: dict) -> dict:
    """
    Triage: Did the agent correctly prioritize urgency?
    Primary metric: urgency_level vs true_risk bracket.
    """
    true_risk = ground_truth["true_risk"]
    urgency = action.get("urgency_level", "medium").lower()

    expected = (
        "low" if true_risk < 0.30
        else "medium" if true_risk < 0.55
        else "high" if true_risk < 0.80
        else "critical"
    )

    urgency_scores = {
        ("low", "low"): 0.92, ("low", "medium"): 0.55, ("low", "high"): 0.20, ("low", "critical"): 0.08,
        ("medium", "low"): 0.55, ("medium", "medium"): 0.90, ("medium", "high"): 0.55, ("medium", "critical"): 0.20,
        ("high", "low"): 0.12, ("high", "medium"): 0.55, ("high", "high"): 0.92, ("high", "critical"): 0.65,
        ("critical", "low"): 0.06, ("critical", "medium"): 0.20, ("critical", "high"): 0.65, ("critical", "critical"): 0.92,
    }

    score = urgency_scores.get((expected, urgency), 0.30)
    risk_score = action.get("risk_score", 0.5)
    detection_bonus = math.exp(-3.0 * (risk_score - true_risk) ** 2) * 0.10

    total = _clamp(score + detection_bonus)
    return {"score": total, "expected_urgency": expected, "got_urgency": urgency}


def grade_risk_stratification(action: dict, ground_truth: dict) -> dict:
    """
    Risk stratification: How accurately did the agent estimate the risk score?
    """
    true_risk = ground_truth["true_risk"]
    predicted_risk = action.get("risk_score", 0.5)

    error = abs(predicted_risk - true_risk)
    base_score = math.exp(-5.0 * error ** 2)

    # Bonus for correct intervention tier
    ideal = ground_truth["ideal_intervention"]
    got = action.get("recommended_intervention", "")
    intervention_bonus = 0.08 if ideal == got else 0.0

    total = _clamp(base_score + intervention_bonus)
    return {
        "score": total,
        "true_risk": true_risk,
        "predicted_risk": predicted_risk,
        "error": round(error, 4),
    }


def grade_early_warning(action: dict, ground_truth: dict) -> dict:
    """
    Early warning: Detect deterioration BEFORE it becomes critical.
    Rewards agents that catch moderate-to-high risk (0.45-0.80) accurately.
    """
    true_risk = ground_truth["true_risk"]
    predicted_risk = action.get("risk_score", 0.5)

    error = abs(predicted_risk - true_risk)

    # In the warning zone, precision is extra important
    if 0.40 <= true_risk <= 0.82:
        penalty_factor = 6.5  # tighter scoring
    else:
        penalty_factor = 3.5

    base_score = math.exp(-penalty_factor * error ** 2)

    # False negative penalty: high risk but predicted low
    if true_risk > 0.65 and predicted_risk < 0.40:
        base_score *= 0.25

    total = _clamp(base_score)
    return {
        "score": total,
        "true_risk": true_risk,
        "predicted_risk": predicted_risk,
        "in_warning_zone": 0.40 <= true_risk <= 0.82,
    }


GRADERS = {
    "triage": grade_triage,
    "risk_stratification": grade_risk_stratification,
    "early_warning": grade_early_warning,
}


def grade(task: str, action: dict, ground_truth: dict) -> dict:
    if task not in GRADERS:
        raise ValueError(f"Unknown task: {task}. Available: {list(GRADERS.keys())}")
    return GRADERS[task](action, ground_truth)
