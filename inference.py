from __future__ import annotations

import asyncio
import json
import os
from typing import Dict, List, Tuple

from openai import OpenAI

from env import Env
from models import Action, TaskId


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _safe_label(value: str, allowed: set[str], default: str) -> str:
    normalized = (value or "").strip().lower()
    return normalized if normalized in allowed else default


def _parse_action_payload(text: str) -> Dict[str, str]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found")
    data = json.loads(text[start : end + 1])
    return {
        "classification": _safe_label(data.get("classification", "normal"), {"spam", "normal", "urgent"}, "normal"),
        "priority": _safe_label(data.get("priority", "medium"), {"low", "medium", "high"}, "medium"),
        "response_text": (data.get("response_text") or "").strip(),
    }


def _fallback(email_subject: str, email_body: str) -> Dict[str, str]:
    text = f"{email_subject} {email_body}".lower()
    if any(k in text for k in ["winner", "crypto", "click", "password", "wire", "bonus"]):
        return {"classification": "spam", "priority": "low", "response_text": ""}
    if any(k in text for k in ["urgent", "p1", "incident", "outage", "immediate", "failed", "cutoff"]):
        return {
            "classification": "urgent",
            "priority": "high",
            "response_text": "Acknowledged. Incident created, mitigation in progress, ETA to follow.",
        }
    return {
        "classification": "normal",
        "priority": "medium",
        "response_text": "Thanks for your email. We will review and respond shortly.",
    }


def _model_action(client: OpenAI, model_name: str, email: Dict[str, str]) -> Dict[str, str]:
    prompt = (
        "Return ONLY JSON with keys classification, priority, response_text. "
        "classification in {spam,normal,urgent}. priority in {low,medium,high}.\n"
        f"Email ID: {email['id']}\n"
        f"Subject: {email['subject']}\n"
        f"Body: {email['body']}\n"
        f"Sender: {email['sender']}\n"
        f"Timestamp: {email['timestamp']}"
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0,
            top_p=1,
            messages=[
                {"role": "system", "content": "You are an enterprise email triage assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content or ""
        return _parse_action_payload(content)
    except Exception:
        return _fallback(email["subject"], email["body"])


async def _run_task(
    env: Env,
    client: OpenAI,
    model_name: str,
    image_name: str,
    task_id: TaskId,
) -> Tuple[float, int]:
    print(f"[START] task={task_id.value} env={image_name} model={model_name}")

    rewards: List[float] = []
    success = True
    steps = 0

    try:
        env.task_id = task_id
        observation = await env.reset()

        for email in observation.emails:
            proposal = _model_action(
                client,
                model_name,
                {
                    "id": email.id,
                    "subject": email.subject,
                    "body": email.body,
                    "sender": email.sender,
                    "timestamp": email.timestamp.isoformat(),
                },
            )
            action = Action(
                email_id=email.id,
                classification=proposal["classification"],
                priority=proposal["priority"],
                response_text=proposal.get("response_text") or None,
            )
            _, reward, done, info = await env.step(action)
            reward_value = float(reward.step_reward)
            rewards.append(reward_value)
            steps += 1

            action_str = json.dumps(action.model_dump(mode="json"), ensure_ascii=True, separators=(",", ":"))
            step_error = info.get("error") if isinstance(info, dict) else None
            step_error_str = json.dumps(step_error) if step_error else "null"
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={steps} action={action_str} reward={reward_value:.2f} "
                f"done={done_str} error={step_error_str}"
            )

            if done:
                break
    except Exception as exc:
        success = False
        if steps == 0:
            print(f"[STEP] step=0 action=null reward=0.00 done=true error={json.dumps(str(exc))}")
        else:
            print(f"[STEP] step={steps + 1} action=null reward=0.00 done=true error={json.dumps(str(exc))}")

    score = float(env.grade()) if success else 0.0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps} score={score:.4f} rewards={rewards_str}")
    return score, steps


async def run_inference() -> Dict[str, float]:
    API_BASE_URL = os.getenv("API_BASE_URL", "").strip() or "https://api.openai.com/v1"
    MODEL_NAME = os.getenv("MODEL_NAME", "").strip() or "gpt-4o-mini"
    HF_TOKEN = _required_env("HF_TOKEN")
    IMAGE_NAME = os.getenv("IMAGE_NAME", "").strip() or "email-triage-openenv"

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    scores: Dict[str, float] = {}

    env = None
    try:
        env = await Env.from_docker_image(IMAGE_NAME)
        for task_id in [TaskId.easy, TaskId.medium, TaskId.hard]:
            score, _steps = await _run_task(env, client, MODEL_NAME, IMAGE_NAME, task_id)
            scores[task_id.value] = round(score, 4)
    except Exception as exc:
        print(f"[START] task=unknown env={IMAGE_NAME} model={MODEL_NAME}")
        print(f"[STEP] step=0 action=null reward=0.00 done=true error={json.dumps(str(exc))}")
        print("[END] success=false steps=0 score=0.0000 rewards=")
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass

    return scores


if __name__ == "__main__":
    try:
        asyncio.run(run_inference())
    except Exception as exc:
        pass
