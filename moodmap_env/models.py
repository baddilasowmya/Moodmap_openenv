from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class BehavioralSignals(BaseModel):
    sleep_hours: float = Field(..., ge=0, le=24)
    activity_steps: int = Field(..., ge=0)
    screen_time_hours: float = Field(..., ge=0, le=24)
    social_interactions: int = Field(..., ge=0)
    heart_rate_variability: float = Field(..., ge=0)
    app_usage_variance: float = Field(..., ge=0, le=1)
    typing_speed_change: float = Field(..., ge=-1, le=1)
    location_entropy: float = Field(..., ge=0, le=1)


class PatientObservation(BaseModel):
    patient_id: str
    timestamp: str
    age: int
    gender: str
    baseline_mood_score: float = Field(..., ge=0, le=1)
    signals: BehavioralSignals
    history_days: int
    prior_episodes: int
    medication_adherence: Optional[float] = None
    task: str
    difficulty: str


class AgentAction(BaseModel):
    patient_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score between 0 and 1")
    recommended_intervention: str
    urgency_level: str
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class StepResult(BaseModel):
    observation: PatientObservation
    action: AgentAction
    reward: float
    reward_breakdown: Dict[str, float]
    done: bool
    info: Dict[str, Any]


class EpisodeResult(BaseModel):
    episode_id: str
    task: str
    difficulty: str
    total_reward: float
    steps: int
    step_results: List[StepResult]
    metadata: Dict[str, Any]
