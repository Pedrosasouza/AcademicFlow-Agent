"""
Configuracao de conexao, gerenciamento de sessoes e inicializacao do banco de dados.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.models import Base, UserModel

load_dotenv()

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@localhost:5432/academic_flow"
)
DEFAULT_USERS: tuple[str, ...] = ("Alice", "Bob", "Charlie")

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_database_url() -> str:
    """Retorna a URL de conexao definida no ambiente ou o padrao local PostgreSQL."""
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine() -> Engine:
    """Retorna (ou cria) a instancia singleton do Engine SQLAlchemy para a URL atual."""
    global _engine, _session_factory

    db_url = get_database_url()
    if _engine is None:
        if db_url == "sqlite:///:memory:":
            _engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        elif db_url.startswith("sqlite"):
            _engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
            )
        else:
            _engine = create_engine(db_url, pool_pre_ping=True)

        _session_factory = sessionmaker(
            bind=_engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _engine


def reset_engine() -> None:
    """Descarta o engine atual para permitir troca de DATABASE_URL (util em testes)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


@contextmanager
def get_db_session() -> Iterator[Session]:
    """Fornece uma sessao transacional com commit automatico e rollback em caso de erro."""
    global _session_factory
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None

    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _ensure_multi_session_columns(engine: Engine) -> None:
    """Garante compatibilidade em bancos locais criados antes da adicao de session_id."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    target_tables = ("tasks", "decisions", "document_sections")

    with engine.begin() as conn:
        for table_name in target_tables:
            if table_name not in existing_tables:
                continue
            cols = {col["name"] for col in inspector.get_columns(table_name)}
            if "session_id" not in cols:
                conn.execute(
                    text(
                        f"ALTER TABLE {table_name} "
                        "ADD COLUMN session_id VARCHAR(100) NOT NULL DEFAULT 'default'"
                    )
                )


def init_db() -> None:
    """Cria as tabelas de forma idempotente e insere os usuarios iniciais caso nao existam."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_multi_session_columns(engine)

    with get_db_session() as session:
        existing_names = set(session.scalars(select(UserModel.name)).all())
        for name in DEFAULT_USERS:
            if name not in existing_names:
                session.add(UserModel(name=name))
