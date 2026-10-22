"""
Modelos ORM com SQLAlchemy 2.0 para persistencia relacional do AcademicFlow-Agent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Classe base declarativa do SQLAlchemy 2.0 para mapeamento das tabelas."""


class UserModel(Base):
    """Participantes cadastrados na sessao colaborativa."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TaskModel(Base):
    """Tarefas do planejamento academico registradas pelo agente."""

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    owner: Mapped[str] = mapped_column(String(100), default="a definir", nullable=False)
    deadline: Mapped[str] = mapped_column(String(100), default="a definir", nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pendente", nullable=False)
    created_at: Mapped[str] = mapped_column(String(20), nullable=False)


class DecisionModel(Base):
    """Decisoes tomadas pelo grupo e suas respectivas justificativas."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[str] = mapped_column(String(20), nullable=False)


class DocumentSectionModel(Base):
    """Secoes estruturadas do documento colaborativo de planejamento."""

    __tablename__ = "document_sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    section: Mapped[str] = mapped_column(String(150), unique=True, index=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(String(20), nullable=False)


class ChatMessageModel(Base):
    """Historico de mensagens trocadas com o agente ReAct."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(100), default="default", index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
