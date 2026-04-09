"""
Deterministic graders for MoodMap tasks.
Each grader takes an action dict and ground_truth dict and returns a score strictly in (0.05, 0.95).
grade() returns a float so the OpenEnv validator can read it directly.
"""
import math

# Hard bounds — strictly between 0 and 1, never 0.0 or 1.0
MIN_SCORE = 0.05
MAX_SCORE = 0.95


def _clamp(v: float) -> float:
    """Clamp to strictly open interval (0, 1) using safe sentinel bounds."""
    return max(MIN_SCORE, min(MAX_SCORE, float(v)))


def _safe_exp_score(error: float, penalty: float) -> float:
    """
    Gaussian decay score. Caps at MAX_SCORE even when error=0
    (which would otherwise produce exp(0)=1.0, an invalid score).
    """
    raw = math.exp(-penalty * error ** 2)
    return _clamp(raw)


def grade_triage(action: dict, ground_truth: dict) -> float:
    """
    Triage: Did the agent correctly prioritize urgency?
    Primary metric: urgency_level vs true_risk bracket.
    Returns a float score strictly in (0.05, 0.95).
    """
    true_risk = float(ground_truth.get("true_risk", 0.5))
    urgency = str(action.get("urgency_level", "medium")).lower().strip()

    # Expected urgency bracket from true risk
    if true_risk < 0.30:
        expected = "low"
    elif true_risk < 0.55:
        expected = "medium"
    elif true_risk < 0.80:
        expected = "high"
    else:
        expected = "critical"

    # All base scores pre-clamped so none reach 0.0 or 1.0
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
        ("critical", "low"):      0.06,
        ("critical", "medium"):   0.18,
        ("critical", "high"):     0.62,
        ("critical", "critical"): 0.88,
    }

    base = urgency_scores.get((expected, urgency), 0.28)

    # Small detection bonus for accurate risk_score (max +0.06 so we never exceed 0.94)
    risk_score = float(action.get("risk_score", 0.5))
    detection_bonus = math.exp(-3.0 * (risk_score - true_risk) ** 2) * 0.06

    return _clamp(base + detection_bonus)


def grade_risk_stratification(action: dict, ground_truth: dict) -> float:
    """
    Risk stratification: How accurately did the agent estimate the risk score?
    Returns a float score strictly in (0.05, 0.95).
    """
    true_risk = float(ground_truth.get("true_risk", 0.5))
    predicted_risk = float(action.get("risk_score", 0.5))

    # Use _safe_exp_score to ensure exp(0)=1.0 never escapes
    error = abs(predicted_risk - true_risk)
    base_score = _safe_exp_score(error, penalty=5.0)

    # Small bonus for correct intervention (max +0.05 stays under 0.95 after clamp)
    ideal = ground_truth.get("ideal_intervention", "")
    got = str(action.get("recommended_intervention", ""))
    intervention_bonus = 0.05 if ideal == got else 0.0

    return _clamp(base_score + intervention_bonus)


def grade_early_warning(action: dict, ground_truth: dict) -> float:
    """
    Early warning: Detect deterioration BEFORE it becomes critical.
    Rewards agents that catch moderate-to-high risk (0.45-0.80) accurately.
    Returns a float score strictly in (0.05, 0.95).
    """
    true_risk = float(ground_truth.get("true_risk", 0.5))
    predicted_risk = float(action.get("risk_score", 0.5))

    error = abs(predicted_risk - true_risk)

    # Tighter scoring in the warning zone
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
    Returns a float strictly in (0.05, 0.95).
    Raises ValueError for unknown tasks.
    """
    if task not in GRADERS:
        raise ValueError(f"Unknown task: '{task}'. Available: {list(GRADERS.keys())}")
    score = GRADERS[task](action, ground_truth)
    # Final safety clamp — guarantees validator never sees 0.0 or 1.0
    return _clamp(float(score))