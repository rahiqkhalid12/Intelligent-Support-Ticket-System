"""
retraining_checks.py

Six independent checks, one per condition from the spec. Each function:
  - takes a DB session and the Thresholds config
  - returns a dict: {key, triggered, severity, message, measured_value, threshold_value}

Keeping them independent (rather than one giant function) means you can
unit test each one, disable one temporarily, or add a 7th check later
without touching the others.
"""

from datetime import datetime, timedelta
import sys
from pathlib import Path

# Make sure backend_connector/ (one level up, where models.py lives) is on
# the import path - needed because this file lives in the mlpos/ subfolder.
_BACKEND_CONNECTOR_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_CONNECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_CONNECTOR_DIR))

from sqlalchemy import func

from models import Ticket, TicketStatus, Response, Feedback
from .retraining_config import RetrainingThresholds


def _window_bounds(recent_days: int, baseline_days: int):
    now = datetime.utcnow()
    recent_start = now - timedelta(days=recent_days)
    baseline_start = recent_start - timedelta(days=baseline_days)
    return baseline_start, recent_start, now


def check_avg_feedback_rating(db, t: RetrainingThresholds) -> dict:
    """Condition 1: average feedback rating in the recent window < threshold."""
    _, recent_start, now = _window_bounds(t.recent_window_days, t.baseline_window_days)

    q = db.query(func.avg(Feedback.rating), func.count(Feedback.id)).filter(
        Feedback.created_at >= recent_start, Feedback.created_at <= now
    )
    avg_rating, count = q.first()

    triggered = (
        count is not None
        and count >= t.min_feedback_sample_size
        and avg_rating is not None
        and avg_rating < t.min_avg_feedback_rating
    )

    return {
        "key": "low_feedback_rating",
        "label": "Average Feedback Rating",
        "triggered": bool(triggered),
        "severity": "critical" if triggered and avg_rating < 2.5 else "warning",
        "measured_value": round(avg_rating, 2) if avg_rating is not None else None,
        "threshold_value": t.min_avg_feedback_rating,
        "sample_size": count,
        "message": (
            f"Average feedback rating dropped to {round(avg_rating, 2)}/5 "
            f"over the last {t.recent_window_days} days (threshold: {t.min_avg_feedback_rating})."
            if triggered else ""
        ),
    }


def check_avg_confidence(db, t: RetrainingThresholds) -> dict:
    """Condition 2: average AI confidence score in the recent window < threshold."""
    _, recent_start, now = _window_bounds(t.recent_window_days, t.baseline_window_days)

    q = db.query(func.avg(Response.confidence_score), func.count(Response.id)).filter(
        Response.created_at >= recent_start, Response.created_at <= now
    )
    avg_conf, count = q.first()

    triggered = (
        count is not None
        and count >= t.min_response_sample_size
        and avg_conf is not None
        and avg_conf < t.min_avg_confidence
    )

    return {
        "key": "low_confidence_score",
        "label": "Average Confidence Score",
        "triggered": bool(triggered),
        "severity": "critical" if triggered and avg_conf < 0.4 else "warning",
        "measured_value": round(avg_conf, 3) if avg_conf is not None else None,
        "threshold_value": t.min_avg_confidence,
        "sample_size": count,
        "message": (
            f"Average AI confidence dropped to {round(avg_conf, 2)} "
            f"over the last {t.recent_window_days} days (threshold: {t.min_avg_confidence})."
            if triggered else ""
        ),
    }


def check_reopened_ratio(db, t: RetrainingThresholds) -> dict:
    """Condition 3: too many reopened tickets in the recent window."""
    _, recent_start, now = _window_bounds(t.recent_window_days, t.baseline_window_days)

    total = db.query(func.count(Ticket.id)).filter(
        Ticket.created_at >= recent_start, Ticket.created_at <= now
    ).scalar() or 0

    reopened = db.query(func.count(Ticket.id)).filter(
        Ticket.created_at >= recent_start,
        Ticket.created_at <= now,
        Ticket.status == TicketStatus.REOPENED,
    ).scalar() or 0

    ratio = (reopened / total) if total else 0.0
    triggered = total >= t.min_ticket_sample_size and ratio > t.max_reopened_ratio

    return {
        "key": "high_reopened_ratio",
        "label": "Reopened Ticket Rate",
        "triggered": bool(triggered),
        "severity": "critical" if triggered and ratio > 0.30 else "warning",
        "measured_value": round(ratio, 3),
        "threshold_value": t.max_reopened_ratio,
        "sample_size": total,
        "message": (
            f"{reopened} of {total} tickets ({round(ratio * 100)}%) were reopened "
            f"in the last {t.recent_window_days} days (threshold: {round(t.max_reopened_ratio * 100)}%)."
            if triggered else ""
        ),
    }


def check_failed_ratio(db, t: RetrainingThresholds) -> dict:
    """Condition 4: too many failed AI responses (tickets marked Failed)."""
    _, recent_start, now = _window_bounds(t.recent_window_days, t.baseline_window_days)

    total = db.query(func.count(Ticket.id)).filter(
        Ticket.created_at >= recent_start, Ticket.created_at <= now
    ).scalar() or 0

    failed = db.query(func.count(Ticket.id)).filter(
        Ticket.created_at >= recent_start,
        Ticket.created_at <= now,
        Ticket.status == TicketStatus.FAILED,
    ).scalar() or 0

    ratio = (failed / total) if total else 0.0
    triggered = total >= t.min_ticket_sample_size and ratio > t.max_failed_ratio

    return {
        "key": "high_failed_ratio",
        "label": "Failed AI Response Rate",
        "triggered": bool(triggered),
        "severity": "critical" if triggered and ratio > 0.35 else "warning",
        "measured_value": round(ratio, 3),
        "threshold_value": t.max_failed_ratio,
        "sample_size": total,
        "message": (
            f"{failed} of {total} tickets ({round(ratio * 100)}%) ended as Failed "
            f"in the last {t.recent_window_days} days (threshold: {round(t.max_failed_ratio * 100)}%)."
            if triggered else ""
        ),
    }


def check_volume_spike(db, t: RetrainingThresholds) -> dict:
    """Condition 5: large increase in ticket volume vs the baseline period."""
    baseline_start, recent_start, now = _window_bounds(t.recent_window_days, t.baseline_window_days)

    recent_count = db.query(func.count(Ticket.id)).filter(
        Ticket.created_at >= recent_start, Ticket.created_at <= now
    ).scalar() or 0

    baseline_count = db.query(func.count(Ticket.id)).filter(
        Ticket.created_at >= baseline_start, Ticket.created_at < recent_start
    ).scalar() or 0

    baseline_daily_avg = (baseline_count / t.baseline_window_days) if t.baseline_window_days else 0
    expected_recent = baseline_daily_avg * t.recent_window_days

    if expected_recent > 0:
        increase_ratio = (recent_count - expected_recent) / expected_recent
    else:
        increase_ratio = 1.0 if recent_count > 0 else 0.0

    triggered = (
        baseline_count >= t.min_ticket_sample_size
        and increase_ratio > t.max_volume_increase_ratio
    )

    return {
        "key": "ticket_volume_spike",
        "label": "Ticket Volume Increase",
        "triggered": bool(triggered),
        "severity": "critical" if triggered and increase_ratio > 1.0 else "warning",
        "measured_value": round(increase_ratio, 3),
        "threshold_value": t.max_volume_increase_ratio,
        "sample_size": recent_count,
        "message": (
            f"Ticket volume is up {round(increase_ratio * 100)}% vs the expected "
            f"baseline in the last {t.recent_window_days} days "
            f"(threshold: +{round(t.max_volume_increase_ratio * 100)}%)."
            if triggered else ""
        ),
    }


def check_new_categories(db, t: RetrainingThresholds) -> dict:
    """Condition 6: new ticket categories (predicted_type) appearing frequently.

    A category counts as "new" if it never appeared during the baseline
    window but shows up at least `new_category_min_occurrences` times in
    the recent window.

    Requires a minimum amount of baseline data first - without that guard,
    an empty/thin baseline (e.g. a brand new system, or the first few
    weeks of data) would make every category look "new" and trigger
    permanently, which isn't a meaningful signal.
    """
    baseline_start, recent_start, now = _window_bounds(t.recent_window_days, t.baseline_window_days)

    baseline_total = db.query(func.count(Ticket.id)).filter(
        Ticket.created_at >= baseline_start, Ticket.created_at < recent_start
    ).scalar() or 0

    if baseline_total < t.min_ticket_sample_size:
        return {
            "key": "new_ticket_categories",
            "label": "New Ticket Categories",
            "triggered": False,
            "severity": "warning",
            "measured_value": 0,
            "threshold_value": t.new_category_min_occurrences,
            "sample_size": baseline_total,
            "new_categories": [],
            "message": "",
        }

    baseline_categories = {
        row[0]
        for row in db.query(Ticket.predicted_type)
        .filter(Ticket.created_at >= baseline_start, Ticket.created_at < recent_start)
        .distinct()
        .all()
        if row[0]
    }

    recent_rows = (
        db.query(Ticket.predicted_type, func.count(Ticket.id))
        .filter(Ticket.created_at >= recent_start, Ticket.created_at <= now)
        .group_by(Ticket.predicted_type)
        .all()
    )

    new_categories = [
        {"category": cat, "count": count}
        for cat, count in recent_rows
        if cat and cat not in baseline_categories and count >= t.new_category_min_occurrences
    ]

    triggered = len(new_categories) > 0

    return {
        "key": "new_ticket_categories",
        "label": "New Ticket Categories",
        "triggered": bool(triggered),
        "severity": "warning",
        "measured_value": len(new_categories),
        "threshold_value": t.new_category_min_occurrences,
        "sample_size": sum(c["count"] for c in new_categories),
        "new_categories": new_categories,
        "message": (
            "New ticket categories appearing frequently: "
            + ", ".join(f"{c['category']} ({c['count']}x)" for c in new_categories)
            if triggered else ""
        ),
    }


ALL_CHECKS = [
    check_avg_feedback_rating,
    check_avg_confidence,
    check_reopened_ratio,
    check_failed_ratio,
    check_volume_spike,
    check_new_categories,
]