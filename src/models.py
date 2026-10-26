"""
Modelos ORM com SQLAlchemy 2.0 para persistencia relacional do AcademicFlow-Agent.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator, TypeEngine, UserDefinedType

EMBEDDING_DIM = 768


class _PgVectorCol(UserDefinedType[list[float]]):
    """Tipo DDL nativo vector(N) para o PostgreSQL com a extensao pgvector."""

    cache_ok = True

    def __init__(self, dimensions: int = EMBEDDING_DIM) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **kw: Any) -> str:
        return f"vector({self.dimensions})"


class PgVectorType(TypeDecorator[list[float]]):
    """Tipo hibrido que usa vector(N) nativo no PostgreSQL e JSON em outros dialetos (SQLite)."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.dimensions = dimensions

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PgVectorCol(self.dimensions))
        return dialect.type_descriptor(JSON())

    def process_bind_param(
        self,
        value: list[float] | None,
        dialect: Dialect,
    ) -> Any:
        if value is None:
            return None
        vec = [float(x) for x in value]
        if dialect.name == "postgresql":
            return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
        return vec

    def process_result_value(
        self,
        value: Any,
        dialect: Dialect,
    ) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip().strip("[]")
            if not cleaned:
                return []
            return [float(x) for x in cleaned.split(",")]
        return [float(x) for x in value]


class TaskStatus(StrEnum):
    """Estados validos no ciclo de vida de uma tarefa do planejamento."""

    PENDING = "pendente"
    IN_PROGRESS = "em_andamento"
    COMPLETED = "concluida"
    CANCELLED = "cancelada"


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
    session_id: Mapped[str] = mapped_column(
        String(100), default="default", index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    owner: Mapped[str] = mapped_column(String(100), default="a definir", nullable=False)
    owner_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    deadline: Mapped[str] = mapped_column(String(100), default="a definir", nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), default=TaskStatus.PENDING.value, nullable=False
    )
    created_at: Mapped[str] = mapped_column(String(20), nullable=False)


class DecisionModel(Base):
    """Decisoes tomadas pelo grupo e suas respectivas justificativas."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(100), default="default", index=True, nullable=False
    )
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[str] = mapped_column(String(20), nullable=False)


class DocumentSectionModel(Base):
    """Secoes estruturadas do documento colaborativo de planejamento."""

    __tablename__ = "document_sections"
    __table_args__ = (
        UniqueConstraint("session_id", "section", name="uq_document_session_section"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(100), default="default", index=True, nullable=False
    )
    section: Mapped[str] = mapped_column(String(150), index=True, nullable=False)
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
    author: Mapped[str | None] = mapped_column(String(100), nullable=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RegulationChunkModel(Base):
    """Trechos (chunks) de editais e normas da universidade indexados com vetores de embedding (RAG)."""

    __tablename__ = "regulation_chunks"
    __table_args__ = (
        Index(
            "ix_regulation_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(150), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        PgVectorType(EMBEDDING_DIM), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
