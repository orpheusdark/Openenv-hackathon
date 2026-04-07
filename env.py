from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

from models import (
    Action,
    Email,
    EnvironmentState,
    GroundTruthEmail,
    Observation,
    Reward,
    TaskId,
)
from tasks.easy import get_task_data as easy_task_data
from tasks.graders import grade_task, response_quality_score
from tasks.hard import get_task_data as hard_task_data
from tasks.medium import get_task_data as medium_task_data


class EmailTriageEnv:
    def __init__(self, task_id: TaskId = TaskId.easy):
        self.task_id = task_id
        self._ground_truth: Dict[str, GroundTruthEmail] = {}
        self._emails: List[Email] = []
        self._processed_flags: Dict[str, bool] = {}
        self._assigned_labels: Dict[str, Dict[str, str]] = {}
        self._action_history: List[Action] = []
        self._cumulative_reward: float = 0.0
        self._current_step: int = 0
        self._done: bool = False
        self._closed: bool = False
        self._max_steps: int = 0
        self._reset_internal()

    def _load_task_rows(self) -> List[dict]:
        if self.task_id == TaskId.easy:
            return easy_task_data()
        if self.task_id == TaskId.medium:
            return medium_task_data()
        return hard_task_data()

    def _build_observation(self) -> Observation:
        return Observation(
            task_id=self.task_id,
            emails=list(self._emails),
            processed_flags=dict(self._processed_flags),
            assigned_labels=dict(self._assigned_labels),
            action_history=list(self._action_history),
            current_step=self._current_step,
            total_emails=len(self._emails),
        )

    def _reset_internal(self) -> Observation:
        rows = self._load_task_rows()
        rows = sorted(rows, key=lambda x: (x["timestamp"], x["id"]))

        self._ground_truth = {}
        self._emails = []
        for row in rows:
            gt = GroundTruthEmail(
                id=row["id"],
                subject=row["subject"],
                body=row["body"],
                sender=row["sender"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                classification=row["classification"],
                priority=row["priority"],
                requires_response=row.get("requires_response", False),
                required_response_keywords=row.get("required_response_keywords", []),
                ideal_response=row.get("ideal_response"),
            )
            self._ground_truth[gt.id] = gt
            self._emails.append(
                Email(
                    id=gt.id,
                    subject=gt.subject,
                    body=gt.body,
                    sender=gt.sender,
                    timestamp=gt.timestamp,
                )
            )

        self._processed_flags = {e.id: False for e in self._emails}
        self._assigned_labels = {}
        self._action_history = []
        self._cumulative_reward = 0.0
        self._current_step = 0
        self._done = False
        self._closed = False
        self._max_steps = min(12, len(self._emails))

        return self._build_observation()

    def _step_internal(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._closed:
            observation = self._build_observation()
            reward = Reward(step_reward=0.0, cumulative_reward=self._cumulative_reward, components={})
            return observation, reward, True, {"message": "Environment is closed."}

        if self._done:
            observation = self._build_observation()
            reward = Reward(step_reward=0.0, cumulative_reward=self._cumulative_reward, components={})
            return observation, reward, True, {"message": "Episode is already done."}

        info: Dict[str, Any] = {}
        components: Dict[str, float] = {
            "classification": 0.0,
            "priority": 0.0,
            "response": 0.0,
            "urgent_ignored_penalty": 0.0,
            "repeat_penalty": 0.0,
        }

        gt = self._ground_truth.get(action.email_id)
        if gt is None:
            components["repeat_penalty"] = -0.1
            step_reward = -0.1
            self._current_step += 1
            self._cumulative_reward += step_reward
            self._done = self._current_step >= self._max_steps
            observation = self._build_observation()
            reward = Reward(
                step_reward=step_reward,
                cumulative_reward=self._cumulative_reward,
                components=components,
            )
            info["error"] = "Unknown email_id"
            return observation, reward, self._done, info

        repeated = self._processed_flags.get(action.email_id, False)
        if repeated:
            components["repeat_penalty"] = -0.1

        if action.classification == gt.classification.value:
            components["classification"] = 0.2

        if self.task_id in (TaskId.medium, TaskId.hard) and action.priority == gt.priority.value:
            components["priority"] = 0.3

        if self.task_id == TaskId.hard and gt.requires_response:
            quality = response_quality_score(action.response_text or "", gt)
            components["response"] = 0.5 * quality

        self._processed_flags[action.email_id] = True
        self._assigned_labels[action.email_id] = {
            "classification": action.classification,
            "priority": action.priority,
        }
        self._action_history.append(action)
        self._current_step += 1

        # Penalize each step where urgent emails remain unprocessed.
        unprocessed_urgent = [
            e for e in self._ground_truth.values() if e.classification.value == "urgent" and not self._processed_flags[e.id]
        ]
        if unprocessed_urgent:
            components["urgent_ignored_penalty"] = -0.2

        step_reward = sum(components.values())
        self._cumulative_reward += step_reward

        all_processed = all(self._processed_flags.values())
        self._done = all_processed or (self._current_step >= self._max_steps)

        observation = self._build_observation()
        reward = Reward(
            step_reward=step_reward,
            cumulative_reward=self._cumulative_reward,
            components=components,
        )
        info["remaining"] = sum(1 for done in self._processed_flags.values() if not done)

        return observation, reward, self._done, info

    def _state_internal(self) -> EnvironmentState:
        return EnvironmentState(
            task_id=self.task_id,
            current_step=self._current_step,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            closed=self._closed,
            processed_flags=dict(self._processed_flags),
            assigned_labels=dict(self._assigned_labels),
            action_history=list(self._action_history),
        )

    def grade(self) -> float:
        return grade_task(self.task_id, self._action_history, self._ground_truth)

    def switch_task(self, task_id: TaskId) -> Observation:
        self.task_id = task_id
        return self._reset_internal()

    async def reset(self) -> Observation:
        return self._reset_internal()

    async def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        return self._step_internal(action)

    async def state(self) -> EnvironmentState:
        return self._state_internal()

    async def close(self) -> None:
        self._closed = True


class Env(EmailTriageEnv):
    @classmethod
    async def from_docker_image(cls, image_name: str, task_id: TaskId = TaskId.easy) -> "Env":
        _ = image_name
        return cls(task_id=task_id)

    @property
    def action_schema(self) -> Dict[str, Any]:
        return {
            "email_id": "str",
            "classification": ["spam", "normal", "urgent"],
            "priority": ["low", "medium", "high"],
            "response_text": "Optional[str]",
        }
