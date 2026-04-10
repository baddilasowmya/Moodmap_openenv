"""
Deterministic graders for MoodMap tasks.
Each grader takes an action dict and ground_truth dict and returns a score strictly in (0.06, 0.94).
grade() returns a float so the OpenEnv validator can read it directly.
"""
import math

# Hard bounds — strictly between 0 and 1, with margin so we never hit the boundary
MIN_SCORE = 0.06
MAX_SCORE = 0.94


def _clamp(v: float) -> float:
    """Clamp to strictly open interval using safe sentinel bounds."""
    return max(MIN_SCORE, min(MAX_SCORE, float(v)))


def _safe_exp_score(error: float, penalty: float) -> float:
    """
    Gaussian decay score. When error=0, exp(0)=1.0 which would be invalid.
    MAX_SCORE=0.94 ensures the clamp always keeps us strictly below 1.0.
    """
    raw = math.exp(-penalty * error ** 2)
    return _clamp(raw)


def grade_triage(action: dict, ground_truth: dict) -> float:
    """
    Triage: Did the agent correctly prioritize urgency?
    Returns a float score strictly in (0, 1).
    """
    true_risk = float(ground_truth.get("true_risk", 0.5))
    urgency = str(action.get("urgency_level", "medium")).lower().strip()

    if true_risk < 0.30:
        expected = "low"
    elif true_risk < 0.55:
        expected = "medium"
    elif true_risk < 0.80:
        expected = "high"
    else:
        expected = "critical"

    urgency_scores = {
        ("low",      "low"):      0.88,
        ("low",      "medium"):   0.55,
        ("low",      "high"):     0.20,
        ("low",      "critical"): 0.08,
        ("medium",   "low"):      0.55,
        ("medium",   "medium"):   0.88,
        ("medium",   "high"):     0.55,
        ("medium",   "critical"): 0.20,
        ("high",     "low"):      0.10,
        ("high",     "medium"):   0.50,
        ("high",     "high"):     0.88,
        ("high",     "critical"): 0.62,
        ("critical", "low"):      0.07,
        ("critical", "medium"):   0.18,
        ("critical", "high"):     0.62,
        ("critical", "critical"): 0.88,
    }

    base = urgency_scores.get((expected, urgency), 0.28)

    # Small detection bonus — max +0.05 so we never exceed 0.93 before clamp
    risk_score = float(action.get("risk_score", 0.5))
    detection_bonus = math.exp(-3.0 * (risk_score - true_risk) ** 2) * 0.05

    return _clamp(base + detection_bonus)


def grade_risk_stratification(action: dict, ground_truth: dict) -> float:
    """
    Risk stratification: How accurately did the agent estimate the risk score?
    Returns a float score strictly in (0, 1).
    """
    true_risk = float(ground_truth.get("true_risk", 0.5))
    predicted_risk = float(action.get("risk_score", 0.5))

    error = abs(predicted_risk - true_risk)
    # penalty=5.0, MAX_SCORE=0.94 means exp(0)=1.0 -> clamped to 0.94, never 1.0
    base_score = _safe_exp_score(error, penalty=5.0)

    # Intervention bonus capped at +0.04 so 0.94+0.04=0.98 -> clamped to 0.94
    ideal = ground_truth.get("ideal_intervention", "")
    got = str(action.get("recommended_intervention", ""))
    intervention_bonus = 0.04 if ideal == got else 0.0

    return _clamp(base_score + intervention_bonus)


def grade_early_warning(action: dict, ground_truth: dict) -> float:
    """
    Early warning: Detect deterioration BEFORE it becomes critical.
    Returns a float score strictly in (0, 1).
    """
    true_risk = float(ground_truth.get("true_risk", 0.5))
    predicted_risk = float(action.get("risk_score", 0.5))

    error = abs(predicted_risk - true_risk)

    penalty_factor = 6.5 if 0.40 <= true_risk <= 0.82 else 3.5
    base_score = _safe_exp_score(error, penalty=penalty_factor)

    # False negative penalty: high risk but predicted low
    if true_risk > 0.65 and predicted_risk < 0.40:
        base_score = _clamp(base_score * 0.25)

    return _clamp(base_score)


GRADERS = {
    "triage":               grade_triage,
    "risk_stratification":  grade_risk_stratification,
    "early_warning":        grade_early_warning,
}


def grade(task: str, action: dict, ground_truth: dict) -> float:
    """
    Main entry point called by OpenEnv validator.
    Returns a float strictly in (0.06, 0.94) — always strictly between 0 and 1.
    """
    if task not in GRADERS:
        raise ValueError(f"Unknown task: '{task}'. Available: {list(GRADERS.keys())}")
    score = GRADERS[task](action, ground_truth)
    return _clamp(float(score))