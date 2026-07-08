"""
retraining_monitor.py

The single entry point the API/scheduler calls. Runs every check, works
out an overall status, and keeps the retraining_alerts table in sync:
  - creates a new alert row the first time a condition trips
  - leaves it alone on subsequent runs while still triggered (no duplicate spam)
  - auto-resolves it once the condition is no longer triggered
"""

from datetime import datetime

from .retraining_checks import ALL_CHECKS
from .retraining_config import THRESHOLDS
from .retraining_models import RetrainingAlert


def run_all_checks(db, thresholds=None) -> list[dict]:
    """Run every check and return their raw results."""
    thresholds = thresholds or THRESHOLDS
    return [check(db, thresholds) for check in ALL_CHECKS]


def sync_alerts(db, results: list[dict]) -> list[RetrainingAlert]:
    for result in results:
        existing = (
            db.query(RetrainingAlert)
            .filter(
                RetrainingAlert.condition_key == result["key"],
                RetrainingAlert.resolved.is_(False),
            )
            .first()
        )

        if result["triggered"] and not existing:
            db.add(
                RetrainingAlert(
                    condition_key=result["key"],
                    message=result["message"],
                    severity=result["severity"],
                    measured_value=result["measured_value"],
                    threshold_value=result["threshold_value"],
                )
            )
        elif result["triggered"] and existing:
            # Still triggered — refresh the numbers so the dashboard
            # shows the current state, not the value from when it first fired.
            existing.message = result["message"]
            existing.severity = result["severity"]
            existing.measured_value = result["measured_value"]
            existing.threshold_value = result["threshold_value"]
        elif not result["triggered"] and existing:
            existing.resolved = True
            existing.resolved_at = datetime.utcnow()

    db.commit()

    return (
        db.query(RetrainingAlert)
        .filter(RetrainingAlert.resolved.is_(False))
        .order_by(RetrainingAlert.created_at.desc())
        .all()
    )


def get_overall_status(results: list[dict]) -> str:
    """OK / warning / retrain_recommended, based on the worst triggered check."""
    triggered = [r for r in results if r["triggered"]]
    if not triggered:
        return "ok"
    if any(r["severity"] == "critical" for r in triggered):
        return "retrain_recommended"
    return "warning"


def evaluate(db, thresholds=None) -> dict:
    """Run checks, sync alerts table, and return a single API-ready payload."""
    results = run_all_checks(db, thresholds)
    active_alerts = sync_alerts(db, results)
    status = get_overall_status(results)

    return {
        "status": status,  # "ok" | "warning" | "retrain_recommended"
        "checked_at": datetime.utcnow().isoformat(),
        "conditions": results,
        "active_alerts": [
            {
                "id": a.id,
                "condition_key": a.condition_key,
                "message": a.message,
                "severity": a.severity,
                "measured_value": a.measured_value,
                "threshold_value": a.threshold_value,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in active_alerts
        ],
    }