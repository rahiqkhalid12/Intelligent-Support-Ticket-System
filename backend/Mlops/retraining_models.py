"""
retraining_models.py

One new table: retraining_alerts. It reuses the same `Base` and `engine`
your app already uses (from database.py), so it lives in the same
support.db file - no separate database needed.

Import `RetrainingAlert` wherever you import your other models (e.g. next
to User/Ticket/Response in models.py, or keep it separate as here - both
work since they share the same Base).
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean

from database import Base


class RetrainingAlert(Base):
    __tablename__ = "retraining_alerts"

    id = Column(Integer, primary_key=True, index=True)

    # Which condition fired, e.g. "low_feedback_rating", "high_failed_ratio"
    condition_key = Column(String, nullable=False)

    # Human-readable message for the dashboard
    message = Column(Text, nullable=False)

    # "warning" or "critical" - lets the dashboard color-code alerts
    severity = Column(String, default="warning")

    # The actual measured value and the threshold it breached, kept as
    # separate columns so the dashboard/API doesn't have to parse the message
    measured_value = Column(Float, nullable=True)
    threshold_value = Column(Float, nullable=True)

    # Whether an admin has acknowledged/dismissed this alert
    resolved = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)