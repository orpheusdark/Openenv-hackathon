from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from env import EmailTriageEnv
from inference import run_inference
from models import Action, TaskId


class ResetRequest(BaseModel):
    task_id: TaskId = TaskId.easy


class GraderRequest(BaseModel):
    task_id: TaskId


app = FastAPI(title="OpenEnv Email Triage & Response", version="1.0.0")
ENV = EmailTriageEnv(task_id=TaskId.easy)


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "service": "email-triage-openenv"}


@app.post("/reset")
async def reset(req: ResetRequest) -> Dict[str, Any]:
    if ENV.task_id != req.task_id:
        ENV.switch_task(req.task_id)
    obs = await ENV.reset()
    return {"observation": obs.model_dump(mode="json")}


@app.post("/step")
async def step(action: Action) -> Dict[str, Any]:
    obs, reward, done, info = await ENV.step(action)
    return {
        "observation": obs.model_dump(mode="json"),
        "reward": reward.model_dump(mode="json"),
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state() -> Dict[str, Any]:
    current_state = await ENV.state()
    return {"state": current_state.model_dump(mode="json")}


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {
        "available_tasks": [t.value for t in TaskId],
        "action_schema": ENV.action_schema,
    }


@app.post("/grader")
def grader(req: GraderRequest) -> Dict[str, float]:
    if ENV.task_id != req.task_id:
        ENV.switch_task(req.task_id)
    return {"score": float(round(ENV.grade(), 6))}


@app.post("/baseline")
async def baseline() -> Dict[str, Any]:
    scores = await run_inference()
    return {"scores": scores}
