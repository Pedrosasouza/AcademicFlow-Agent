"""Esquema relacional inicial versionado (usuarios, tarefas, decisoes, secoes, chat e editais RAG).

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2026-10-26 16:25:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

from src.models import EMBEDDING_DIM, PgVectorType

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Cria as 6 tabelas relacionais com indices, extensao pgvector e indice HNSW."""
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    if is_postgres:
        op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_name"), "users", ["name"], unique=True)

    op.create_table(
        "tasks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False, server_default="default"),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("owner", sa.String(length=100), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=True),
        sa.Column("deadline", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.String(length=20), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_tasks_session_id"), "tasks", ["session_id"], unique=False)
    op.create_index(op.f("ix_tasks_owner_id"), "tasks", ["owner_id"], unique=False)

    op.create_table(
        "decisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False, server_default="default"),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=20), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_decisions_session_id"), "decisions", ["session_id"], unique=False
    )

    op.create_table(
        "document_sections",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False, server_default="default"),
        sa.Column("section", sa.String(length=150), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.String(length=20), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "section", name="uq_document_session_section"),
    )
    op.create_index(
        op.f("ix_document_sections_session_id"),
        "document_sections",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_sections_section"),
        "document_sections",
        ["section"],
        unique=False,
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False, server_default="default"),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("author", sa.String(length=100), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_calls", sa.JSON(), nullable=True),
        sa.Column("tool_call_id", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_chat_messages_session_id"),
        "chat_messages",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chat_messages_user_id"),
        "chat_messages",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "regulation_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=150), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", PgVectorType(EMBEDDING_DIM), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_regulation_chunks_source"),
        "regulation_chunks",
        ["source"],
        unique=False,
    )
    if is_postgres:
        op.execute(
            sa.text(
                "CREATE INDEX IF NOT EXISTS ix_regulation_chunks_embedding_hnsw "
                "ON regulation_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )


def downgrade() -> None:
    """Reverte a criacao de todas as tabelas na ordem inversa."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("DROP INDEX IF EXISTS ix_regulation_chunks_embedding_hnsw"))
    op.drop_index(op.f("ix_regulation_chunks_source"), table_name="regulation_chunks")
    op.drop_table("regulation_chunks")

    op.drop_index(op.f("ix_chat_messages_user_id"), table_name="chat_messages")
    op.drop_index(op.f("ix_chat_messages_session_id"), table_name="chat_messages")
    op.drop_table("chat_messages")

    op.drop_index(op.f("ix_document_sections_section"), table_name="document_sections")
    op.drop_index(op.f("ix_document_sections_session_id"), table_name="document_sections")
    op.drop_table("document_sections")

    op.drop_index(op.f("ix_decisions_session_id"), table_name="decisions")
    op.drop_table("decisions")

    op.drop_index(op.f("ix_tasks_owner_id"), table_name="tasks")
    op.drop_index(op.f("ix_tasks_session_id"), table_name="tasks")
    op.drop_table("tasks")

    op.drop_index(op.f("ix_users_name"), table_name="users")
    op.drop_table("users")
