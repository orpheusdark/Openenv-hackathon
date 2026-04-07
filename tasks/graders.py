from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Dict, List

from models import Action, GroundTruthEmail, TaskId


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _tokenize(text: str) -> list[str]:
    clean = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in text)
    return [t for t in clean.split() if t]


def _cosine_similarity(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0

    va = Counter(ta)
    vb = Counter(tb)
    common = set(va) & set(vb)
    dot = float(sum(va[t] * vb[t] for t in common))
    na = sqrt(float(sum(v * v for v in va.values())))
    nb = sqrt(float(sum(v * v for v in vb.values())))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def response_quality_score(response_text: str, gt: GroundTruthEmail) -> float:
    if not gt.requires_response:
        return 1.0

    response = (response_text or "").strip().lower()
    if not response:
        return 0.0

    keywords = [k.lower() for k in gt.required_response_keywords]
    if keywords:
        keyword_hits = sum(1 for k in keywords if k in response)
        keyword_score = keyword_hits / len(keywords)
    else:
        keyword_score = 1.0

    ideal = (gt.ideal_response or "").strip()
    semantic_score = _cosine_similarity(response, ideal) if ideal else 0.5

    missing_penalty = 0.0
    if keywords:
        missing_ratio = 1.0 - keyword_score
        missing_penalty = 0.35 * missing_ratio

    raw = (0.6 * keyword_score) + (0.4 * semantic_score) - missing_penalty
    return max(0.0, min(1.0, raw))


def grade_easy(actions: List[Action], gt_by_id: Dict[str, GroundTruthEmail]) -> float:
    if not gt_by_id:
        return 0.0

    correct = 0
    for action in actions:
        gt = gt_by_id.get(action.email_id)
        if not gt:
            continue
        if action.classification == gt.classification:
            correct += 1

    return max(0.0, min(1.0, _safe_div(correct, len(gt_by_id))))


def grade_medium(actions: List[Action], gt_by_id: Dict[str, GroundTruthEmail]) -> float:
    if not gt_by_id:
        return 0.0

    weight_total = 0.0
    score_total = 0.0

    for gt in gt_by_id.values():
        weight = 2.0 if gt.classification.value == "urgent" else 1.0
        weight_total += weight

    action_map = {a.email_id: a for a in actions}

    for email_id, gt in gt_by_id.items():
        action = action_map.get(email_id)
        weight = 2.0 if gt.classification.value == "urgent" else 1.0
        if action is None:
            continue

        cls = 1.0 if action.classification == gt.classification else 0.0
        prio = 1.0 if action.priority == gt.priority else 0.0
        score_total += weight * ((0.6 * cls) + (0.4 * prio))

    return max(0.0, min(1.0, _safe_div(score_total, weight_total)))


def grade_hard(actions: List[Action], gt_by_id: Dict[str, GroundTruthEmail]) -> float:
    if not gt_by_id:
        return 0.0

    action_map = {a.email_id: a for a in actions}

    cls_total = 0.0
    prio_total = 0.0
    resp_total = 0.0
    total = float(len(gt_by_id))

    for email_id, gt in gt_by_id.items():
        action = action_map.get(email_id)
        if action is None:
            continue

        cls_total += 1.0 if action.classification == gt.classification else 0.0
        prio_total += 1.0 if action.priority == gt.priority else 0.0
        resp_total += response_quality_score(action.response_text or "", gt)

    cls_score = _safe_div(cls_total, total)
    prio_score = _safe_div(prio_total, total)
    resp_score = _safe_div(resp_total, total)

    combined = (0.4 * cls_score) + (0.3 * prio_score) + (0.3 * resp_score)
    return max(0.0, min(1.0, combined))


def grade_task(task_id: TaskId, actions: List[Action], gt_by_id: Dict[str, GroundTruthEmail]) -> float:
    if task_id == TaskId.easy:
        return grade_easy(actions, gt_by_id)
    if task_id == TaskId.medium:
        return grade_medium(actions, gt_by_id)
    return grade_hard(actions, gt_by_id)
