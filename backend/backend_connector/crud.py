from sqlalchemy import String, cast, func
from sqlalchemy.orm import selectinload

from models import User, Ticket, Response, Feedback, Log, TicketStatus


# ======================================================
# USER CRUD
# ======================================================

def get_user_by_email(db, email: str):
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db, user_id: int):
    return db.query(User).filter(User.id == user_id).first()


def get_total_users(db):
    return db.query(User).filter(User.role == "client").count()


def create_user(
    db,
    name: str,
    email: str,
    password_hash: str,
    role: str = "client"
):
    user = User(
        name=name,
        email=email,
        password_hash=password_hash,
        role=role
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def update_user(
    db,
    user_id: int,
    name: str = None,
    email: str = None,
    password_hash: str = None,
    role: str = None
):
    user = get_user_by_id(db, user_id)

    if not user:
        return None

    if name is not None:
        user.name = name
    if email is not None:
        user.email = email
    if password_hash is not None:
        user.password_hash = password_hash
    if role is not None:
        user.role = role

    db.commit()
    db.refresh(user)

    return user


def delete_user(db, user_id: int):
    user = get_user_by_id(db, user_id)

    if not user:
        return False

    db.delete(user)
    db.commit()

    return True


def get_all_users(db):
    return db.query(User).order_by(User.created_at.desc()).all()


# ======================================================
# TICKET CRUD
# ======================================================

def create_ticket(
    db,
    subject,
    description,
    predicted_type,
    predicted_priority,
    predicted_queue,
    status=TicketStatus.OPEN,
    user_id=None
):
    """Create a new support ticket, optionally tied to a user."""

    ticket = Ticket(
        user_id=user_id,
        subject=subject,
        description=description,
        predicted_type=predicted_type,
        predicted_priority=predicted_priority,
        predicted_queue=predicted_queue,
        status=status
    )

    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    return ticket


def get_ticket(db, ticket_id: int):
    return db.query(Ticket).filter(Ticket.id == ticket_id).first()


def get_ticket_with_responses(db, ticket_id: int):
    """Fetch a single ticket with its responses eagerly loaded
    (used to reload saved AI responses after a page refresh)."""
    return (
        db.query(Ticket)
        .options(selectinload(Ticket.responses))
        .filter(Ticket.id == ticket_id)
        .first()
    )


def get_all_tickets(db):
    """Admin ticket list: every ticket joined with the submitting user's info."""
    return (
        db.query(Ticket, User.name.label("user_name"), User.email.label("user_email"))
        .outerjoin(User, Ticket.user_id == User.id)
        .order_by(Ticket.created_at.desc())
        .all()
    )


def get_admin_ticket_list(db, status: str = None):
    """Admin ticket list with optional status filter."""
    query = (
        db.query(Ticket, User.name.label("user_name"), User.email.label("user_email"))
        .outerjoin(User, Ticket.user_id == User.id)
    )

    if status:
        query = query.filter(Ticket.status == status)

    return query.order_by(Ticket.created_at.desc()).all()


def get_tickets_by_user(db, user_id: int):
    """Client ticket history: all tickets a given user has submitted,
    most recent first, with their AI responses preloaded."""
    return (
        db.query(Ticket)
        .options(selectinload(Ticket.responses))
        .filter(Ticket.user_id == user_id)
        .order_by(Ticket.created_at.desc())
        .all()
    )


# Alias for clarity per spec naming
def get_client_ticket_history(db, user_id: int):
    return get_tickets_by_user(db, user_id)


def get_reopened_tickets(db):
    """All tickets currently in the Reopened state."""
    return (
        db.query(Ticket)
        .filter(Ticket.status == TicketStatus.REOPENED)
        .order_by(Ticket.created_at.desc())
        .all()
    )


def get_failed_tickets(db):
    """All tickets currently in the Failed state."""
    return (
        db.query(Ticket)
        .filter(Ticket.status == TicketStatus.FAILED)
        .order_by(Ticket.created_at.desc())
        .all()
    )


def update_ticket_status(db, ticket_id: int, status: str):
    """Update a ticket's status. Accepts either a TicketStatus enum member
    or a raw string (e.g. 'Resolved', 'Reopened')."""

    ticket = get_ticket(db, ticket_id)

    if not ticket:
        return None

    if isinstance(status, TicketStatus):
        ticket.status = status
    else:
        try:
            ticket.status = TicketStatus(status)
        except ValueError:
            raise ValueError(
                f"Invalid ticket status '{status}'. "
                f"Must be one of: {[s.value for s in TicketStatus]}"
            )

    db.commit()
    db.refresh(ticket)

    return ticket


def delete_ticket(db, ticket_id: int):
    ticket = get_ticket(db, ticket_id)

    if not ticket:
        return False

    db.delete(ticket)
    db.commit()

    return True


def set_ticket_admin_response(db, ticket_id: int, admin_response: str):
    """Save a manually-written admin reply onto a ticket (separate from
    the AI-generated Response rows)."""
    ticket = get_ticket(db, ticket_id)

    if not ticket:
        return None

    ticket.admin_response = admin_response
    db.commit()
    db.refresh(ticket)

    return ticket


# ======================================================
# RESPONSE CRUD
# ======================================================

def create_response(
    db,
    ticket_id,
    generated_response,
    confidence_score=None,
    model_used="RAG",
    response_time_ms=None
):
    """Persist an AI-generated response for a ticket."""

    response = Response(
        ticket_id=ticket_id,
        generated_response=generated_response,
        confidence_score=confidence_score,
        model_used=model_used,
        response_time_ms=response_time_ms
    )

    db.add(response)
    db.commit()
    db.refresh(response)

    return response


def save_ai_response(
    db,
    ticket_id,
    generated_response,
    confidence_score=None,
    model_used="RAG",
    response_time_ms=None
):
    """Automatically save an AI response right after it's generated.
    Thin wrapper around create_response for readability at call sites."""
    return create_response(
        db,
        ticket_id=ticket_id,
        generated_response=generated_response,
        confidence_score=confidence_score,
        model_used=model_used,
        response_time_ms=response_time_ms
    )


def get_response(db, response_id: int):
    return db.query(Response).filter(Response.id == response_id).first()


def get_responses_for_ticket(db, ticket_id: int):
    """All saved AI responses for a ticket, oldest first — used to
    reload the conversation/response history after a page refresh."""
    return (
        db.query(Response)
        .filter(Response.ticket_id == ticket_id)
        .order_by(Response.created_at.asc())
        .all()
    )


def get_latest_response_for_ticket(db, ticket_id: int):
    """Most recent AI response for a ticket."""
    return (
        db.query(Response)
        .filter(Response.ticket_id == ticket_id)
        .order_by(Response.created_at.desc())
        .first()
    )


# ======================================================
# FEEDBACK CRUD
# ======================================================

def create_feedback(
    db,
    response_id: int,
    user_id: int,
    rating: int,
    comment: str = None
):
    feedback = Feedback(
        response_id=response_id,
        user_id=user_id,
        rating=rating,
        comment=comment
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return feedback


def get_feedback_for_response(db, response_id: int):
    return db.query(Feedback).filter(
        Feedback.response_id == response_id
    ).all()


def get_all_feedback(db):
    return (
        db.query(
            Feedback,
            User.name.label("user_name"),
            User.email.label("user_email"),
            Response.ticket_id.label("ticket_id"),
        )
        .outerjoin(User, Feedback.user_id == User.id)
        .outerjoin(Response, Feedback.response_id == Response.id)
        .order_by(Feedback.created_at.desc())
        .all()
    )


def get_feedback_by_id(db, feedback_id: int):
    return db.query(Feedback).filter(Feedback.id == feedback_id).first()


def delete_feedback(db, feedback_id: int):
    feedback = get_feedback_by_id(db, feedback_id)

    if not feedback:
        return False

    db.delete(feedback)
    db.commit()

    return True


# ======================================================
# LOG CRUD
# ======================================================

def create_log(
    db,
    ticket_id: int,
    admin_id: int,
    action: str
):
    log = Log(
        ticket_id=ticket_id,
        admin_id=admin_id,
        action=action
    )

    db.add(log)
    db.commit()
    db.refresh(log)

    return log


def get_logs(db):
    return db.query(Log).order_by(Log.created_at.desc()).all()


def get_all_logs(db):
    return (
        db.query(Log, User.name.label("user_name"), User.email.label("user_email"))
        .outerjoin(User, Log.admin_id == User.id)
        .order_by(Log.created_at.desc())
        .all()
    )


def get_ticket_logs(db, ticket_id: int):
    return (
        db.query(Log)
        .filter(Log.ticket_id == ticket_id)
        .order_by(Log.created_at.asc())
        .all()
    )


# ======================================================
# DASHBOARD STATISTICS
# ======================================================

def get_dashboard_stats(db):
    """Aggregate counts used to populate an admin dashboard."""

    total_tickets = db.query(Ticket).count()
    total_users = get_total_users(db)
    total_feedback = db.query(Feedback).count()

    status_counts = dict(
        db.query(cast(Ticket.status, String), func.count(Ticket.id))
        .group_by(cast(Ticket.status, String))
        .all()
    )

    # Normalize keys to plain strings and ensure every status is present
    status_summary = {s.value: 0 for s in TicketStatus}
    for status, count in status_counts.items():
        key = status.value if isinstance(status, TicketStatus) else status
        status_summary[key] = count

    avg_rating = db.query(func.avg(Feedback.rating)).scalar()
    avg_confidence = db.query(func.avg(Response.confidence_score)).scalar()
    avg_response_time_ms = db.query(func.avg(Response.response_time_ms)).scalar()

    return {
        "total_tickets": total_tickets,
        "total_users": total_users,
        "total_feedback": total_feedback,
        "tickets_by_status": status_summary,
        "open_tickets": status_summary.get(TicketStatus.OPEN.value, 0),
        "resolved_tickets": status_summary.get(TicketStatus.RESOLVED.value, 0),
        "reopened_tickets": status_summary.get(TicketStatus.REOPENED.value, 0),
        "failed_tickets": status_summary.get(TicketStatus.FAILED.value, 0),
        "in_progress_tickets": status_summary.get(TicketStatus.IN_PROGRESS.value, 0),
        "average_feedback_rating": round(avg_rating, 2) if avg_rating else None,
        "average_confidence_score": round(avg_confidence, 2) if avg_confidence else None,
        "average_response_time_ms": round(avg_response_time_ms, 2) if avg_response_time_ms else None,
    }


# ======================================================
# MONITORING DASHBOARD (Person B)
# ======================================================

def get_ticket_priority_distribution(db):
    """Count of tickets grouped by predicted priority (High/Medium/Low).

    Groups on the raw DB value first (so tickets with different casing,
    e.g. "high" vs "High", are still counted correctly), then normalizes
    the label casing afterward so it lines up with the frontend's color
    map (PRIORITY_COLORS = {High, Medium, Low, Unknown}). Without this
    normalization, mismatched casing would make chart slices fall back
    to gray instead of their intended color.
    """
    rows = (
        db.query(Ticket.predicted_priority, func.count(Ticket.id))
        .group_by(Ticket.predicted_priority)
        .all()
    )
    distribution = {}
    for priority, count in rows:
        label = priority.strip().capitalize() if priority and priority.strip() else "Unknown"
        distribution[label] = distribution.get(label, 0) + count
    return distribution


def get_ticket_queue_distribution(db):
    """Count of tickets grouped by predicted queue/department."""
    rows = (
        db.query(Ticket.predicted_queue, func.count(Ticket.id))
        .group_by(Ticket.predicted_queue)
        .all()
    )
    return {(queue or "Unknown"): count for queue, count in rows}


def get_recent_activity(db, limit: int = 20):
    """Most recent admin/system log entries, for the activity feed."""
    rows = (
        db.query(Log, User.name.label("admin_name"))
        .outerjoin(User, Log.admin_id == User.id)
        .order_by(Log.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": log.id,
            "ticket_id": log.ticket_id,
            "admin_name": admin_name,
            "action": log.action,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log, admin_name in rows
    ]