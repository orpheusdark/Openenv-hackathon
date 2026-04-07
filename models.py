from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TaskId(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class Classification(str, Enum):
    spam = "spam"
    normal = "normal"
    urgent = "urgent"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    timestamp: datetime


class Action(BaseModel):
    email_id: str
    classification: Literal["spam", "normal", "urgent"]
    priority: Literal["low", "medium", "high"]
    response_text: Optional[str] = None


class Reward(BaseModel):
    step_reward: float
    cumulative_reward: float
    components: Dict[str, float] = Field(default_factory=dict)


class Observation(BaseModel):
    task_id: TaskId
    emails: List[Email]
    processed_flags: Dict[str, bool]
    assigned_labels: Dict[str, Dict[str, str]]
    action_history: List[Action]
    current_step: int
    total_emails: int


class GroundTruthEmail(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    timestamp: datetime
    classification: Classification
    priority: Priority
    requires_response: bool = False
    required_response_keywords: List[str] = Field(default_factory=list)
    ideal_response: Optional[str] = None


class EnvironmentState(BaseModel):
    task_id: TaskId
    current_step: int
    cumulative_reward: float
    done: bool
    closed: bool
    processed_flags: Dict[str, bool]
    assigned_labels: Dict[str, Dict[str, str]]
    action_history: List[Action]
