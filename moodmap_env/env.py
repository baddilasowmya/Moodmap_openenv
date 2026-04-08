import uuid
import math
from datetime import datetime
from typing import Optional
from .models import AgentAction, StepResult, EpisodeResult
from .data import generate_patient, DIFFICULTY_CONFIG, INTERVENTIONS

INTERVENTION_RANK = {
    "no_action": 0,
    "psychoeducation_materials": 1,
    "peer_support_referral": 2,
    "therapist_consultation": 3,
    "medication_review": 3,
    "wellness_check": 3,
    "crisis_hotline_referral": 4,
    "emergency_services": 5,
}

IDEAL_INTERVENTION_BY_RISK = [
    (0.25, "no_action"),
    (0.45, "psychoeducation_materials"),
    (0.60, "peer_support_referral"),
    (0.75, "therapist_consultation"),
    (0.88, "crisis_hotline_referral"),
    (1.01, "emergency_services"),
]


def _clamp_reward(r: float) -> float:
    """Ensure reward is strictly in (0.05, 0.95) — never 0.0 or 1.0"""
    return max(0.05, min(0.95, r))


def _detection_score(predicted_risk: float, true_risk: float) -> float:
    """How close is the predicted risk to the true risk?"""
    error = abs(predicted_risk - true_risk)
    # Gaussian decay — small errors get high scores
    raw = math.exp(-4.0 * error ** 2)
    return raw


def _intervention_score(action: AgentAction, true_risk: float) -> float:
    """Is the recommended intervention appropriate for the risk level?"""
    predicted = action.recommended_intervention
    if predicted not in INTERVENTION_RANK:
        return 0.1

    # Determine ideal for true risk
    ideal = "no_action"
    for threshold, name in IDEAL_INTERVENTION_BY_RISK:
        if true_risk < threshold:
            ideal = name
            break

    pred_rank = INTERVENTION_RANK[predicted]
    ideal_rank = INTERVENTION_RANK[ideal]
    rank_diff = abs(pred_rank - ideal_rank)

    # Perfect = 1.0, off by 1 = 0.65, off by 2 = 0.35, off by 3+ = 0.1
    scores = {0: 1.0, 1: 0.65, 2: 0.35}
    return scores.get(rank_diff, 0.1)


def _timeliness_score(action: AgentAction, true_risk: float) -> float:
    """Urgency should match risk. High risk + low urgency = penalized."""
    urgency_map = {"low": 0.15, "medium": 0.5, "high": 0.85, "critical": 0.98}
    predicted_urgency = urgency_map.get(action.urgency_level.lower(), 0.5)
    error = abs(predicted_urgency - true_risk)
    return math.exp(-3.0 * error ** 2)


def _harm_avoidance_score(action: AgentAction, true_risk: float) -> float:
    """
    Penalize dangerous errors:
    - Recommending no_action for high-risk patients (false negative)
    - Recommending emergency services for low-risk patients (over-escalation)
    """
    pred_rank = INTERVENTION_RANK.get(action.recommended_intervention, 0)

    # False negative: high risk (>0.7) but low intervention (rank <= 1)
    if true_risk > 0.70 and pred_rank <= 1:
        return 0.05  # severe penalty

    # Over-escalation: low risk (<0.3) but emergency intervention (rank >= 4)
    if true_risk < 0.30 and pred_rank >= 4:
        return 0.15  # moderate penalty

    # Under-treatment: moderate risk (0.5-0.85) + no action
    if 0.50 < true_risk < 0.85 and pred_rank == 0:
        return 0.10

    return 0.90  # no harm


def compute_reward(action: AgentAction, ground_truth: dict, task: str, difficulty: str) -> dict:
    true_risk = ground_truth["true_risk"]

    detection = _detection_score(action.risk_score, true_risk)
    intervention = _intervention_score(action, true_risk)
    timeliness = _timeliness_score(action, true_risk)
    harm = _harm_avoidance_score(action, true_risk)

    # Weights vary by task
    weights = {
        "triage": {"detection": 0.35, "intervention": 0.30, "timeliness": 0.25, "harm": 0.10},
        "risk_stratification": {"detection": 0.40, "intervention": 0.25, "timeliness": 0.20, "harm": 0.15},
        "early_warning": {"detection": 0.25, "intervention": 0.25, "timeliness": 0.30, "harm": 0.20},
    }.get(task, {"detection": 0.30, "intervention": 0.30, "timeliness": 0.20, "harm": 0.20})

    # Difficulty multiplier (harder tasks have tighter scoring)
    diff_factor = {"easy": 1.0, "medium": 0.95, "hard": 0.88}.get(difficulty, 0.95)

    raw = (
        weights["detection"] * detection
        + weights["intervention"] * intervention
        + weights["timeliness"] * timeliness
        + weights["harm"] * harm
    ) * diff_factor

    total = _clamp_reward(raw)

    return {
        "total": round(total, 4),
        "detection": round(_clamp_reward(detection), 4),
        "intervention": round(_clamp_reward(intervention), 4),
        "timeliness": round(_clamp_reward(timeliness), 4),
        "harm_avoidance": round(_clamp_reward(harm), 4),
        "true_risk": true_risk,
        "ideal_intervention": ground_truth["ideal_intervention"],
    }


class MoodMapEnv:
    """
    OpenEnv-compatible environment for passive mental health monitoring.
    
    Tasks:
      - triage: Prioritize patients by urgency (easy)
      - risk_stratification: Classify risk tier from behavioral signals (medium)
      - early_warning: Detect early deterioration from subtle signal changes (hard)
    """

    TASKS = ["triage", "risk_stratification", "early_warning"]
    DIFFICULTIES = ["easy", "medium", "hard"]

    def __init__(self, task: str = "triage", difficulty: str = "easy", max_steps: int = 5):
        assert task in self.TASKS, f"task must be one of {self.TASKS}"
        assert difficulty in self.DIFFICULTIES, f"difficulty must be one of {self.DIFFICULTIES}"
        self.task = task
        self.difficulty = difficulty
        self.max_steps = max_steps
        self._episode_id = None
        self._step = 0
        self._current_patient = None
        self._history = []

    def reset(self) -> dict:
        self._episode_id = f"EP-{uuid.uuid4().hex[:10].upper()}"
        self._step = 0
        self._history = []
        self._current_patient = generate_patient(task=self.task, difficulty=self.difficulty)
        return {
            "episode_id": self._episode_id,
            "observation": self._current_patient["observation"].model_dump(),
            "task": self.task,
            "difficulty": self.difficulty,
            "step": 0,
            "max_steps": self.max_steps,
        }

    def step(self, action: AgentAction) -> dict:
        if self._current_patient is None:
            raise RuntimeError("Call reset() before step()")

        ground_truth = self._current_patient["ground_truth"]
        observation = self._current_patient["observation"]

        reward_breakdown = compute_reward(action, ground_truth, self.task, self.difficulty)
        total_reward = reward_breakdown.pop("total")

        self._step += 1
        done = self._step >= self.max_steps

        result = {
            "episode_id": self._episode_id,
            "step": self._step,
            "observation": observation.model_dump(),
            "action": action.model_dump(),
            "reward": total_reward,
            "reward_breakdown": reward_breakdown,
            "done": done,
            "info": {
                "task": self.task,
                "difficulty": self.difficulty,
                "true_risk_revealed": ground_truth["true_risk"] if done else None,
                "ideal_intervention_revealed": ground_truth["ideal_intervention"] if done else None,
            },
        }

        self._history.append(result)

        # Generate next patient if not done
        if not done:
            self._current_patient = generate_patient(task=self.task, difficulty=self.difficulty)
            result["next_observation"] = self._current_patient["observation"].model_dump()

        return result

    def get_episode_summary(self) -> dict:
        if not self._history:
            return {}
        total = sum(s["reward"] for s in self._history)
        return {
            "episode_id": self._episode_id,
            "task": self.task,
            "difficulty": self.difficulty,
            "total_reward": round(total, 4),
            "mean_reward": round(total / len(self._history), 4),
            "steps": len(self._history),
            "step_rewards": [s["reward"] for s in self._history],
        }
