---
title: OpenEnv Email Triage & Response Environment
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# OpenEnv Email Triage & Response Environment

## Environment Purpose

This project implements a production-style OpenEnv environment for enterprise inbox triage. The environment simulates realistic operations, compliance, security, and customer communication workflows where an agent must classify incoming email, prioritize it, and generate responses when required.

## Real-World Relevance

Modern support, operations, and incident-response teams process mixed inbound email streams including spam/phishing, routine business communication, and high-severity escalations. Incorrect triage causes SLA breaches, delayed incident response, and compliance risk. This environment captures those constraints in a deterministic benchmark.

## Observation Schema

Observation is a typed Pydantic model containing:

- `task_id`: active task (`easy`, `medium`, `hard`)
- `emails`: list of emails (`id`, `subject`, `body`, `sender`, `timestamp`)
- `processed_flags`: per-email completion flags
- `assigned_labels`: predicted classification/priority per email
- `action_history`: full action trace
- `current_step`: step counter
- `total_emails`: total emails in task episode

## Action Schema

Action is a typed Pydantic model:

- `email_id`: target email
- `classification`: `spam` | `normal` | `urgent`
- `priority`: `low` | `medium` | `high`
- `response_text`: optional response draft

## Reward Logic

Dense reward is computed each step:

- `+0.2` correct classification
- `+0.3` correct priority (medium/hard tasks)
- `+0.5 * response_quality` (hard task, partial scoring)
- `-0.2` when urgent emails remain unprocessed after a step
- `-0.1` repeated action on already-processed email

Cumulative reward is tracked across the episode.

## Tasks

### Easy

- Goal: classification only
- Dataset: 12 realistic business emails
- Grader: classification accuracy in `[0.0, 1.0]`

### Medium

- Goal: classification + prioritization
- Dataset: 12 realistic business emails
- Grader: weighted accuracy; urgent mistakes carry higher weight

### Hard

- Goal: classification + prioritization + response generation
- Dataset: 12 realistic business emails requiring mixed response behavior
- Grader includes:
  - keyword matching
  - rule-based semantic similarity (cosine over token counts)
  - penalty for missing required response information

## Determinism and Episode Control

- Task datasets are static and sorted deterministically by timestamp + ID
- `reset()` always returns clean state
- `step()` is deterministic for given input action sequence
- Episode termination occurs when all emails are processed or max steps is reached

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

## API Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `POST /baseline` (runs `inference.py` internally)

## Example API Usage

```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"easy"}'
```

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"email_id":"EZ-001","classification":"spam","priority":"low","response_text":null}'
```

## Inference Script

Run the required script:

```bash
python inference.py
```

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `IMAGE_NAME`

The inference flow is async-compatible and uses:

```python
env = await Env.from_docker_image(IMAGE_NAME)
observation = await env.reset()
observation, reward, done, info = await env.step(action)
await env.close()
```

The script prints strict logs:

- `[START]` with task name
- repeated `[STEP]` blocks (action/observation/reward)
- `[END]` with final score

## Docker

Build and run:

```bash
docker build -t email-triage-openenv .
docker run --rm -p 7860:7860 email-triage-openenv
```

HF Spaces compatibility is provided by using a standard FastAPI container and listening on `0.0.0.0:7860`.

## Expected Baseline Score Range

Because inference relies on external model behavior, exact values vary by model endpoint. Typical deterministic run ranges:

- easy: `0.70 - 1.00`
- medium: `0.60 - 0.95`
- hard: `0.45 - 0.85`

These ranges assume a capable instruction-following model with `temperature=0`.
