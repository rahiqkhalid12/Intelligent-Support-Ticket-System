import enum

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    Enum as SAEnum,
)
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base


# ==========================
# Enums
# ==========================

class TicketStatus(str, enum.Enum):
    OPEN = "Open"
    RESOLVED = "Resolved"
    REOPENED = "Reopened"
    FAILED = "Failed"
    IN_PROGRESS = "In Progress"


# ==========================
# Users
# ==========================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False)

    email = Column(String, unique=True, nullable=False, index=True)

    password_hash = Column(String, nullable=False)

    role = Column(String, default="client")  # client / admin

    created_at = Column(DateTime, default=datetime.utcnow)

    # One user -> many tickets
    tickets = relationship(
        "Ticket", back_populates="user", cascade="all, delete-orphan"
    )

    # One user -> many feedback records they submitted
    feedbacks = relationship(
        "Feedback", back_populates="user", cascade="all, delete-orphan"
    )

    # One user (as admin) -> many logs
    logs = relationship(
        "Log", back_populates="admin", cascade="all, delete-orphan"
    )


# ==========================
# Tickets
# ==========================

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    subject = Column(String, nullable=True)

    description = Column(Text, nullable=False)

    # --- prediction labels ---
    predicted_type = Column(String)

    predicted_priority = Column(String)

    predicted_queue = Column(String)

    # --- status ---
    status = Column(
        SAEnum(
            TicketStatus,
            native_enum=False,
            length=20,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=TicketStatus.OPEN,
        nullable=False,
    )

    # --- manual admin reply (separate from the AI-generated Response rows) ---
    admin_response = Column(Text, nullable=True)

    # --- timestamps ---
    created_at = Column(DateTime, default=datetime.utcnow)

    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    user = relationship("User", back_populates="tickets")

    # One ticket -> many responses
    responses = relationship(
        "Response", back_populates="ticket", cascade="all, delete-orphan"
    )

    # One ticket -> many logs
    logs = relationship(
        "Log", back_populates="ticket", cascade="all, delete-orphan"
    )


# ==========================
# AI Responses
# ==========================

class Response(Base):
    __tablename__ = "responses"

    id = Column(Integer, primary_key=True, index=True)

    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False)

    generated_response = Column(Text)

    confidence_score = Column(Float, nullable=True)

    model_used = Column(String, nullable=True)

    response_time_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    ticket = relationship("Ticket", back_populates="responses")

    # One response -> many feedback records
    feedbacks = relationship(
        "Feedback", back_populates="response", cascade="all, delete-orphan"
    )


# ==========================
# Feedback
# ==========================

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)

    response_id = Column(Integer, ForeignKey("responses.id"), nullable=False)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    rating = Column(Integer)  # 1-5 stars

    comment = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    response = relationship("Response", back_populates="feedbacks")

    user = relationship("User", back_populates="feedbacks")


# ==========================
# Logs (admin actions on tickets)
# ==========================

class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)

    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False)

    admin_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    action = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)

    ticket = relationship("Ticket", back_populates="logs")

    admin = relationship("User", back_populates="logs")