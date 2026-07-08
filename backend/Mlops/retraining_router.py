"""
retraining_router.py

Wire this into your main FastAPI app with:

    from retraining_router import router as retraining_router
    app.include_router(retraining_router)

Endpoints (assumes your existing admin routes are already auth-protected -
add your existing admin auth dependency to these the same way you do
elsewhere, e.g. Depends(get_current_admin)):

    GET  /admin/monitoring/retraining
         -> full status payload: overall status, all 6 condition results,
            and currently active alerts. This is what the dashboard polls.

    GET  /admin/monitoring/retraining/alerts
         -> alert history (active + resolved), for an "alert log" view.

    POST /admin/monitoring/retraining/alerts/{alert_id}/resolve
         -> manually dismiss/acknowledge an alert.

    POST /admin/monitoring/retraining/check
         -> force a fresh check right now (e.g. a "Run Check" button).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from .retraining_models import RetrainingAlert
from .retraining_monitor import evaluate

router = APIRouter(prefix="/admin/monitoring/retraining", tags=["retraining-monitoring"])


@router.get("")
def get_retraining_status(db: Session = Depends(get_db)):
    """Main endpoint for the dashboard: runs checks fresh every call."""
    return evaluate(db)


@router.post("/check")
def force_check(db: Session = Depends(get_db)):
    """Same as GET, exposed as POST for an explicit 'Run Check Now' button."""
    return evaluate(db)


@router.get("/alerts")
def get_alert_history(resolved: bool | None = None, db: Session = Depends(get_db)):
    """Alert history. ?resolved=true / ?resolved=false to filter, omit for all."""
    query = db.query(RetrainingAlert)
    if resolved is not None:
        query = query.filter(RetrainingAlert.resolved == resolved)

    alerts = query.order_by(RetrainingAlert.created_at.desc()).all()

    return [
        {
            "id": a.id,
            "condition_key": a.condition_key,
            "message": a.message,
            "severity": a.severity,
            "measured_value": a.measured_value,
            "threshold_value": a.threshold_value,
            "resolved": a.resolved,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
        }
        for a in alerts
    ]


@router.post("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    """Manually acknowledge/dismiss an alert from the dashboard."""
    from datetime import datetime

    alert = db.query(RetrainingAlert).filter(RetrainingAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.resolved = True
    alert.resolved_at = datetime.utcnow()
    db.commit()
    db.refresh(alert)

    return {"id": alert.id, "resolved": alert.resolved}