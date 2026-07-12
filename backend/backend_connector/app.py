from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from typing import Optional
from pathlib import Path
import sys
import requests
import time
from jose import JWTError, jwt
import datetime

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from crud import (
    create_user,
    get_user_by_email,
    create_ticket,
    create_response,
    get_tickets_by_user,
    get_all_tickets,
    get_all_feedback,
    get_all_logs,
    get_total_users,
    update_ticket_status,
    get_response,
    create_feedback,
    create_log,
    set_ticket_admin_response,   # NEW: needed for the admin manual-reply endpoint
    get_dashboard_stats,               # NEW: monitoring dashboard
    get_ticket_priority_distribution,  # NEW: monitoring dashboard
    get_ticket_queue_distribution,     # NEW: monitoring dashboard
    get_recent_activity,               # NEW: monitoring dashboard
)
from auth import hash_password, verify_password

# Always use the database stored beside this file, regardless of the folder
# from which Uvicorn is started.
DATABASE_PATH = Path(__file__).resolve().with_name("support.db")
engine = create_engine(
    f"sqlite:///{DATABASE_PATH.as_posix()}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ================================================================
# Config
# ================================================================
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM  = "HS256"
TOKEN_EXPIRE_HOURS = 24

AZURE_PREDICT_URL = os.getenv("AZURE_PREDICT_URL")

app = FastAPI(title="AI Support System API")

# Allow frontend (HTML files opened in browser) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# Helpers
# ================================================================
def create_token(user_id: int, email: str, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please log in again.")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token.")


# FastAPI dependency for auth header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security

security = HTTPBearer()

def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db),
):
    payload = decode_token(credentials.credentials)
    user = get_user_by_email(db, payload["email"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


def require_admin(user=Depends(require_auth)):
    if (user.role or "").strip().casefold() != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


from Mlops.retraining_router import router as retraining_router
from Mlops.retraining_models import RetrainingAlert
from database import get_db as retraining_get_db

RetrainingAlert.__table__.create(bind=engine, checkfirst=True)
app.dependency_overrides[retraining_get_db] = get_db
app.include_router(retraining_router, dependencies=[Depends(require_admin)])


def response_confidence(prediction: dict, similar_tickets: list) -> float:
    """Use model confidence when supplied; otherwise use RAG similarity.

    The currently deployed classifier only returns labels. Azure AI Search's
    vector score is therefore the best available signal for response quality.
    """
    raw = prediction.get("confidence_score", prediction.get("confidence"))
    if raw is None and similar_tickets:
        raw = max((ticket.get("score") or 0) for ticket in similar_tickets)
    try:
        return round(max(0.0, min(1.0, float(raw))), 4)
    except (TypeError, ValueError):
        return 0.0

def process_ticket_ai(ticket_id: int, user_id: int, text: str):
    """Runs in the background after the ticket is created. Does the slow
    work (Azure classification, retrieval, generation) and updates the
    ticket + response once done, however long that takes."""
    db = SessionLocal()
    try:
        start = time.perf_counter()

        # ── Azure ML classification ──
        try:
            az_res = requests.post(AZURE_PREDICT_URL, json={"text": text}, timeout=120)
            az_res.raise_for_status()
            prediction = az_res.json().get("prediction", {})
        except Exception as e:
            print(f"Classification error (ticket {ticket_id}): {e}")
            update_ticket_status(db, ticket_id, "Failed")
            create_log(db, ticket_id=ticket_id, admin_id=user_id, action="Classification failed")
            return

        # ── Save predictions onto the ticket now that we have them ──
        from models import Ticket
        ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if ticket:
            ticket.predicted_type     = prediction.get("type")
            ticket.predicted_priority = prediction.get("priority")
            ticket.predicted_queue    = prediction.get("queue")
            db.commit()
        create_log(db, ticket_id=ticket_id, admin_id=user_id, action="Prediction completed")

        # ── Azure Cognitive Search retrieval ──
        similar_tickets = []
        try:
            from retrieve_azure import retrieve_similar_tickets
            similar_tickets = retrieve_similar_tickets(text, top_k=3)
        except Exception as e:
            print(f"Retrieval warning (ticket {ticket_id}): {e}")

        # ── Qwen response generation ──
        generated_response = ""
        try:
            from qwen_service import generate_response
            generated_response = generate_response(text, prediction, similar_tickets)
        except Exception as e:
            print(f"Generation warning (ticket {ticket_id}): {e}")
            generated_response = "Our team will get back to you shortly."

        processing_ms = int((time.perf_counter() - start) * 1000)
        confidence = response_confidence(prediction, similar_tickets)

        create_response(
            db=db,
            ticket_id=ticket_id,
            generated_response=generated_response,
            confidence_score=confidence,
            model_used="RAG",
            response_time_ms=processing_ms,
        )
        create_log(db, ticket_id=ticket_id, admin_id=user_id, action="AI response generated")

        generation_failed = (
            not generated_response.strip()
            or generated_response in {
                "Unable to generate AI response at the moment.",
                "Our team will get back to you shortly.",
            }
        )
        final_status = "Failed" if generation_failed else "Resolved"
        update_ticket_status(db, ticket_id, final_status)
        create_log(
            db,
            ticket_id=ticket_id,
            admin_id=user_id,
            action="Ticket failed" if generation_failed else "Ticket automatically resolved",
        )
    finally:
        db.close()
# ================================================================
# Schemas
# ================================================================
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str
    role: str          # "user" or "admin" — frontend sends which button was clicked

class TicketRequest(BaseModel):
    subject: Optional[str] = None
    description: str

class StatusUpdate(BaseModel):
    status: str

class FeedbackRequest(BaseModel):
    response_id: int
    rating: int
    comment: Optional[str] = None

class AdminReplyRequest(BaseModel):
    response: str


# Allowed ticket statuses under the new lifecycle. "Open" is no longer part
# of the workflow: tickets start "In Progress" while AI classification/
# generation is running, then move to "Resolved" or "Failed" automatically,
# and can only reach "Reopened" via the client's reopen endpoint.
ALLOWED_STATUSES = {"In Progress", "Resolved", "Failed", "Reopened"}


# ================================================================
# Health check
# ================================================================
@app.get("/")
def home():
    return {"message": "AI Support System API is running"}


# ================================================================
# Auth endpoints  (called by main.html)
# ================================================================

@app.post("/auth/signup")
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    """Register a new user account. Role is always 'user' on signup —
    admin accounts are created manually in the database."""
    existing = get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    user = create_user(
        db=db,
        name=body.name,
        email=body.email,
        password_hash=hash_password(body.password),
        role="client",
    )

    token = create_token(user.id, user.email, user.role)

    return {
        "token": token,
        "role":  user.role,
        "name":  user.name,
    }


@app.post("/auth/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Login endpoint. The frontend sends role='user' or role='admin'
    depending on which button was clicked. We verify:
    1. Email exists
    2. Password is correct
    3. User's actual DB role matches the requested role
    If the role doesn't match, return a clear error (e.g. a regular
    user who clicks 'Sign In as Admin' gets rejected here)."""

    user = get_user_by_email(db, body.email)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # Check role matches what button was clicked
    requested_role = body.role.strip().casefold()
    actual_role = (user.role or "client").strip().casefold()

    if requested_role not in {"user", "admin"}:
        raise HTTPException(status_code=400, detail="Invalid login role.")

    if requested_role == "admin" and actual_role != "admin":
        raise HTTPException(
            status_code=403,
            detail="You do not have admin access."
        )

    if requested_role == "user" and actual_role == "admin":
        raise HTTPException(
            status_code=403,
            detail="Please use 'Sign In as Admin' for your account."
        )

    token = create_token(user.id, user.email, user.role)

    return {
        "token": token,
        "role":  user.role,
        "name":  user.name,
    }


# ================================================================
# Ticket endpoints  (called by client_dashboard.html)
# ================================================================

@app.post("/tickets")
@app.post("/tickets")
def submit_ticket(
    body: TicketRequest,
    background_tasks: BackgroundTasks,
    user=Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Submit a ticket. Returns immediately with status "In Progress";
    classification + retrieval + generation happen in the background so
    the client never waits on (or times out from) the AI services."""

    text = (body.subject or "") + " " + body.description

    ticket = create_ticket(
        db=db,
        user_id=user.id,
        subject=body.subject,
        description=body.description,
        predicted_type=None,
        predicted_priority=None,
        predicted_queue=None,
        status="In Progress",
    )
    create_log(db, ticket_id=ticket.id, admin_id=user.id, action="Ticket created")

    background_tasks.add_task(process_ticket_ai, ticket.id, user.id, text.strip())

    return {
        "status": "success",
        "ticket_id": ticket.id,
        "ticket_status": "In Progress",
    }


@app.patch("/tickets/{ticket_id}/reopen")
def reopen_ticket(
    ticket_id: int,
    user=Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Client reopens their own resolved ticket. Only the ticket owner may
    do this, and only tickets currently "Resolved" are eligible."""

    ticket = next((item for item in get_tickets_by_user(db, user.id) if item.id == ticket_id), None)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    if ticket.status != "Resolved":
        raise HTTPException(status_code=409, detail="Only resolved tickets can be reopened.")

    update_ticket_status(db, ticket.id, "Reopened")
    create_log(db, ticket_id=ticket.id, admin_id=user.id, action="Customer reopened ticket")

    return {"status": "success", "ticket_id": ticket.id, "new_status": "Reopened"}


@app.post("/feedback")
def submit_feedback(
    body: FeedbackRequest,
    user=Depends(require_auth),
    db: Session = Depends(get_db),
):
    if body.rating < 1 or body.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")
    response = get_response(db, body.response_id)
    if not response or not response.ticket or response.ticket.user_id != user.id:
        raise HTTPException(status_code=404, detail="AI response not found.")

    feedback = create_feedback(
        db=db,
        response_id=response.id,
        user_id=user.id,
        rating=body.rating,
        comment=(body.comment or "").strip() or None,
    )
    create_log(
        db,
        ticket_id=response.ticket_id,
        admin_id=user.id,
        action=f"Customer submitted feedback ({body.rating}/5)",
    )

    return {"status": "success", "feedback_id": feedback.id, "ticket_id": response.ticket_id}


@app.get("/tickets/my")
def my_tickets(
    user=Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Return the logged-in user's ticket history, including both the
    AI-generated response(s) and any manual admin reply, so the client
    dashboard can display both side by side."""
    tickets = get_tickets_by_user(db, user.id)
    return {
        "tickets": [
            {
                "id": ticket.id,
                "subject": ticket.subject,
                "description": ticket.description,
                "predicted_type": ticket.predicted_type,
                "predicted_priority": ticket.predicted_priority,
                "predicted_queue": ticket.predicted_queue,
                "status": ticket.status,
                "admin_response": ticket.admin_response,
                "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
                "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
                "responses": [
                    {
                        "id": response.id,
                        "ticket_id": response.ticket_id,
                        "generated_response": response.generated_response,
                        "confidence_score": response.confidence_score,
                        "model_used": response.model_used,
                        "response_time_ms": response.response_time_ms,
                        "created_at": response.created_at.isoformat() if response.created_at else None,
                    }
                    for response in sorted(ticket.responses, key=lambda item: item.created_at, reverse=True)
                ],
            }
            for ticket in tickets
        ]
    }


# ================================================================
# Admin endpoints  (called by admin_dashboard.html)
# ================================================================

@app.get("/admin/tickets")
def admin_all_tickets(
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Return EVERY ticket (all statuses), including a `needs_attention`
    flag so the frontend can filter its "Tickets Needing Attention" table
    down to Reopened / Failed / In Progress, while Resolved tickets stay
    available in the payload for dashboard statistics."""
    ticket_rows = get_all_tickets(db)
    total_users = get_total_users(db)
    needs_attention_statuses = {"Reopened", "Failed", "In Progress"}
    tickets = [
        {
            **{column.name: getattr(ticket, column.name) for column in ticket.__table__.columns},
            "user_name": user_name,
            "user_email": user_email,
            "needs_attention": ticket.status in needs_attention_statuses,
        }
        for ticket, user_name, user_email in ticket_rows
    ]
    return {"tickets": tickets, "total_users": total_users}


@app.get("/admin/feedback")
def admin_all_feedback(
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    rows = get_all_feedback(db)
    feedback = [
        {
            **{column.name: getattr(item, column.name) for column in item.__table__.columns},
            "user_name": user_name,
            "user_email": user_email,
            "ticket_id": ticket_id,
        }
        for item, user_name, user_email, ticket_id in rows
    ]
    return {"feedback": feedback}


@app.get("/admin/logs")
def admin_all_logs(
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    rows = get_all_logs(db)
    logs = [
        {
            **{column.name: getattr(item, column.name) for column in item.__table__.columns},
            "user_name": user_name,
            "user_email": user_email,
        }
        for item, user_name, user_email in rows
    ]
    return {"logs": logs}


@app.get("/admin/monitoring")
def admin_monitoring(
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Aggregated KPIs + distributions + recent activity for the
    monitoring tab inside admin_dashboard.html."""
    stats = get_dashboard_stats(db)
    return {
        **stats,
        "priority_distribution": get_ticket_priority_distribution(db),
        "queue_distribution": get_ticket_queue_distribution(db),
        "recent_activity": get_recent_activity(db, limit=20),
    }


@app.patch("/admin/tickets/{ticket_id}/status")
def admin_update_status(
    ticket_id: int,
    body: StatusUpdate,
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    if body.status not in ALLOWED_STATUSES:
        raise HTTPException(status_code=400, detail=f"Status must be one of {ALLOWED_STATUSES}")

    updated = update_ticket_status(db, ticket_id, body.status)
    if not updated:
        raise HTTPException(status_code=404, detail="Ticket not found.")

    create_log(db, ticket_id=ticket_id, admin_id=user.id, action="status_changed")

    return {"status": "success", "ticket_id": ticket_id, "new_status": body.status}


@app.patch("/admin/tickets/{ticket_id}/reply")
def admin_reply(
    ticket_id: int,
    body: AdminReplyRequest,
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin writes a manual reply to a ticket. This is saved separately
    from the AI-generated response, and immediately resolves the ticket."""

    if not body.response.strip():
        raise HTTPException(status_code=400, detail="Response cannot be empty.")

    ticket = set_ticket_admin_response(db, ticket_id, body.response.strip())
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")

    create_log(db, ticket_id=ticket.id, admin_id=user.id, action="Admin replied")

    update_ticket_status(db, ticket.id, "Resolved")
    create_log(db, ticket_id=ticket.id, admin_id=user.id, action="Admin resolved ticket")

    return {
        "status": "success",
        "ticket_id": ticket.id,
        "admin_response": ticket.admin_response,
        "new_status": "Resolved",
    }