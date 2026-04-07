from __future__ import annotations

from typing import Any, Dict

from fastapi import Body, FastAPI

from env import Env
from inference import run_inference
from models import Action, TaskId

app = FastAPI(title="OpenEnv Email Triage & Response", version="1.0.0")
ENV = Env(task_id=TaskId.easy)


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "service": "email-triage-openenv"}


@app.post("/reset")
async def reset(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    task_value = payload.get("task_id", TaskId.easy)
    task_id = TaskId(task_value) if not isinstance(task_value, TaskId) else task_value
    if ENV.task_id != task_id:
        ENV.switch_task(task_id)
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
def grader(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, float]:
    task_value = payload.get("task_id", ENV.task_id)
    task_id = TaskId(task_value) if not isinstance(task_value, TaskId) else task_value
    if ENV.task_id != task_id:
        ENV.switch_task(task_id)
    return {"score": float(round(ENV.grade(), 6))}


@app.post("/baseline")
async def baseline() -> Dict[str, Any]:
    scores = await run_inference()
    return {"scores": scores}


def run() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    run()
