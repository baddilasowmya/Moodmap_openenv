"""
MoodMap Passive Mental Health — OpenEnv server using official framework.

Uses create_app() from openenv-core so all required endpoints are
auto-registered: /reset /step /state /schema /metadata /health /ws /mcp
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid
import math
from typing import Any, Dict, Optional

from pydantic import Field, ConfigDict

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    Action,
    Observation,
    State,
    EnvironmentMetadata,
)
from openenv.core.env_server.http_server import create_app

from moodmap_env import MoodMapEnv
from moodmap_env.models import AgentAction as _AgentAction
from graders import grade

# ── Score bounds — must match graders/__init__.py and openenv.yaml ──────────
MIN_SCORE = 0.06
MAX_SCORE = 0.94


def _clamp(v: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, float(v)))


# ── Action model ─────────────────────────────────────────────────────────────

class MoodMapAction(Action):
    """Action submitted by the agent for a patient assessment."""

    model_config = ConfigDict(extra="allow")

    patient_id: str = Field(default="unknown", description="Patient identifier")
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Predicted risk score (0.06–0.94)")
    recommended_intervention: str = Field(default="no_action", description="Recommended intervention")
    urgency_level: str = Field(default="medium", description="Urgency: low/medium/high/critical")
    reasoning: str = Field(default="", description="Agent reasoning")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in assessment")


# ── Observation model ─────────────────────────────────────────────────────────

class MoodMapObservation(Observation):
    """Observation returned to the agent after each step."""

    model_config = ConfigDict(extra="allow")

    patient_data: Dict[str, Any] = Field(default_factory=dict, description="Patient behavioral signals")
    task: str = Field(default="triage", description="Current task")
    difficulty: str = Field(default="easy", description="Task difficulty")
    episode_id: str = Field(default="", description="Episode identifier")
    step: int = Field(default=0, description="Current step number")


# ── Environment ───────────────────────────────────────────────────────────────

class MoodMapEnvironment(Environment):
    """
    Passive mental health monitoring environment.

    Tasks:
      triage              — Prioritize patients by urgency (easy)
      risk_stratification — Classify risk tier from signals (medium)
      early_warning       — Detect early deterioration (hard)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._env: Optional[MoodMapEnv] = None
        self._task: str = "triage"
        self._difficulty: str = "easy"

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MoodMapObservation:
        task = str(kwargs.get("task", "triage")).lower().strip()
        difficulty = str(kwargs.get("difficulty", "easy")).lower().strip()

        valid_tasks = ["triage", "risk_stratification", "early_warning"]
        valid_diffs = ["easy", "medium", "hard"]
        if task not in valid_tasks:
            task = "triage"
        if difficulty not in valid_diffs:
            difficulty = "easy"

        self._task = task
        self._difficulty = difficulty
        self._env = MoodMapEnv(task=task, difficulty=difficulty, max_steps=5)
        result = self._env.reset()

        ep_id = episode_id or result["episode_id"]
        self._state = State(episode_id=ep_id, step_count=0)

        return MoodMapObservation(
            patient_data=result["observation"],
            task=task,
            difficulty=difficulty,
            episode_id=ep_id,
            step=0,
            done=False,
            reward=None,
        )

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(
        self,
        action: MoodMapAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MoodMapObservation:
        if self._env is None:
            self.reset()

        # Clamp floats to safe range
        risk_score = _clamp(float(action.risk_score))
        confidence = _clamp(float(action.confidence))

        # Normalise urgency
        urgency = str(action.urgency_level).lower().strip()
        if urgency not in {"low", "medium", "high", "critical"}:
            urgency = "medium"

        agent_action = _AgentAction(
            patient_id=str(action.patient_id),
            risk_score=risk_score,
            recommended_intervention=str(action.recommended_intervention),
            urgency_level=urgency,
            reasoning=str(action.reasoning),
            confidence=confidence,
        )

        result = self._env.step(agent_action)

        # Compute grader score
        grader_score = _clamp(float(grade(
            task=self._env.task,
            action=agent_action.model_dump(),
            ground_truth={
                "true_risk": result["reward_breakdown"].get("true_risk", risk_score),
                "ideal_intervention": result["reward_breakdown"].get(
                    "ideal_intervention", action.recommended_intervention
                ),
            },
        )))

        reward = _clamp(float(result["reward"]))
        self._state.step_count += 1

        next_obs = result.get("next_observation", result["observation"])

        return MoodMapObservation(
            patient_data=next_obs,
            task=self._env.task,
            difficulty=self._env.difficulty,
            episode_id=result["episode_id"],
            step=result["step"],
            done=result["done"],
            reward=reward,
            metadata={
                "grader_score": grader_score,
                "reward_breakdown": result["reward_breakdown"],
            },
        )

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="MoodMap Passive Mental Health Environment",
            description=(
                "An RL environment where an LLM agent analyzes behavioral signals "
                "(sleep, activity, screen time, HRV, social interactions) collected "
                "passively from smartphones and wearables to assess patient risk and "
                "recommend appropriate interventions."
            ),
            version="1.0.0",
            author="MoodMap Team",
        )


# ── App ───────────────────────────────────────────────────────────────────────
# create_app registers: /reset /step /state /schema /metadata /health /ws /mcp

app = create_app(
    MoodMapEnvironment,
    MoodMapAction,
    MoodMapObservation,
    env_name="moodmap",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()