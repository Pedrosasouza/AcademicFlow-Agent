"""
Configuracao de conexao, gerenciamento de sessoes e inicializacao do banco de dados.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, select
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


def init_db() -> None:
    """Cria as tabelas de forma idempotente e insere os usuarios iniciais caso nao existam."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    with get_db_session() as session:
        existing_names = set(session.scalars(select(UserModel.name)).all())
        for name in DEFAULT_USERS:
            if name not in existing_names:
                session.add(UserModel(name=name))
