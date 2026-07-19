"""
3.5 Adaptive Scoring
Rule-based (not ML) by design — there's no training data yet. Every
decision is logged (topic, correctness, hints used, resulting change)
so that log can become training data for a real ML model later.
"""
import json
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "decision_log.jsonl")

LEVELS = ["beginner", "intermediate", "advanced"]


class StudentHistory:
    """Tracks a student's recent streaks. Swap the in-memory store below
    for a real DB table keyed by student_id in production."""

    def __init__(self):
        self.correct_streak = 0
        self.wrong_in_a_row = 0


_HISTORY: dict[str, StudentHistory] = {}  # student_id -> StudentHistory


def _get_history(student_id: str) -> StudentHistory:
    if student_id not in _HISTORY:
        _HISTORY[student_id] = StudentHistory()
    return _HISTORY[student_id]


def _bump_level(current_level: str, action: str) -> str:
    idx = LEVELS.index(current_level) if current_level in LEVELS else 1
    if action == "increase":
        idx = min(idx + 1, len(LEVELS) - 1)
    elif action == "decrease":
        idx = max(idx - 1, 0)
    return LEVELS[idx]


def update_difficulty(history: StudentHistory) -> str:
    """Pure rule-based decision, exactly per the guide's spec."""
    if history.correct_streak >= 3:
        return "increase"
    if history.wrong_in_a_row >= 2:
        return "decrease"
    return "maintain"


def _log_decision(entry: dict) -> None:
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def record_score_update(
    student_id: str,
    topic: str,
    correct: bool,
    hints_used: int,
    current_level: str = "intermediate",
) -> dict:
    """
    Updates streak counters, applies the rule-based decision, updates
    mastery level, and logs everything for future ML training.
    Returns {"newMasteryLevel": str, "action": str}.
    """
    history = _get_history(student_id)

    if correct:
        history.correct_streak += 1
        history.wrong_in_a_row = 0
    else:
        history.wrong_in_a_row += 1
        history.correct_streak = 0

    action = update_difficulty(history)
    new_level = _bump_level(current_level, action)

    _log_decision({
        "student_id": student_id,
        "topic": topic,
        "correct": correct,
        "hints_used": hints_used,
        "correct_streak": history.correct_streak,
        "wrong_in_a_row": history.wrong_in_a_row,
        "previous_level": current_level,
        "action": action,
        "new_level": new_level,
    })

    return {"newMasteryLevel": new_level, "action": action}
