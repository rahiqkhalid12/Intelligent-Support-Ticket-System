"""
retraining_config.py

All tunable numbers for the retraining monitoring system live here.
Nothing else in the module should hardcode a threshold - if you want to
change sensitivity, change it here only.
"""

from dataclasses import dataclass


@dataclass
class RetrainingThresholds:
    # 1. Average feedback rating (Feedback.rating, 1-5) in the recent window
    #    falls below this -> triggers.
    min_avg_feedback_rating: float = 3.5

    # 2. Average AI confidence (Response.confidence_score, 0-1) in the
    #    recent window falls below this -> triggers.
    min_avg_confidence: float = 0.65

    # 3. Reopened ticket ratio (Reopened / total tickets) in the recent
    #    window exceeds this -> triggers.
    max_reopened_ratio: float = 0.15  # 15%

    # 4. Failed ticket ratio (Failed / total tickets) in the recent
    #    window exceeds this -> triggers.
    max_failed_ratio: float = 0.20  # 20%

    # 5. Ticket volume increase: (recent_count - baseline_daily_avg * recent_days)
    #    relative increase over baseline exceeds this -> triggers.
    max_volume_increase_ratio: float = 0.50  # +50%

    # 6. A ticket category (predicted_type) that did NOT appear in the
    #    baseline window, appears at least this many times in the recent
    #    window -> triggers (signals a genuinely new kind of request).
    new_category_min_occurrences: int = 5

    # Rolling windows used by every check above.
    recent_window_days: int = 7
    baseline_window_days: int = 30

    # Minimum data volume before a check is allowed to fire at all - avoids
    # false alarms from a handful of records (e.g. 1 bad rating on day one).
    min_feedback_sample_size: int = 5
    min_response_sample_size: int = 5
    min_ticket_sample_size: int = 5


THRESHOLDS = RetrainingThresholds()