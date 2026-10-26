"""
Configuracao de conexao, gerenciamento de sessoes e inicializacao do banco de dados.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import math
import os
import re

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.models import Base, RegulationChunkModel, UserModel

load_dotenv()

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@localhost:5432/academic_flow"
)
DEFAULT_USERS: tuple[str, ...] = ("Alice", "Bob", "Charlie")

DEFAULT_REGULATIONS: tuple[tuple[str, str, str], ...] = (
    (
        "Regimento de Espacos ICMC-USP",
        "Reserva de Auditorio e Salas",
        "A solicitacao de reserva do auditorio principal para eventos academicos exige antecedencia minima de 30 dias corridos no sistema institucional e aprovacao da Comissao de Cultura e Extensao.",
    ),
    (
        "Edital PRCEU-USP 01/2026",
        "Orcamento e Prestacao de Contas de Coffee Break",
        "Despesas com coffee break e alimentacao sao limitadas a 20% do valor total aprovado no edital, exigindo emissao de nota fiscal com CNPJ da universidade e lista de presenca assinada.",
    ),
    (
        "Normas de Extensao Universitaria USP",
        "Emissao de Certificados e Frequencia Minima",
        "A emissao de certificados oficiais para participantes e ouvintes exige frequencia minima comprovada de 75% da carga horaria total do evento e cadastro previo no sistema Apolo.",
    ),
    (
        "Diretrizes de Simposios e Semanas Academicas",
        "Submissao de Artigos e Comite Cientifico",
        "Eventos com apresentacao de trabalhos cientificos devem publicar o edital de submissao com no minimo 45 dias de antecedencia e contar com avaliacao duplo-cega por docentes ou pos-graduandos.",
    ),
)

_DOMAIN_CONCEPTS: tuple[tuple[str, ...], ...] = (
    ("auditorio", "reserva", "sala", "espaco", "antecedencia", "30", "local", "capacidade"),
    ("coffee", "break", "alimentacao", "orcamento", "verba", "nota", "fiscal", "prestacao", "contas", "custo"),
    ("certificado", "certificados", "frequencia", "presenca", "75", "carga", "horaria", "apolo", "ouvinte"),
    ("artigo", "artigos", "submissao", "trabalho", "trabalhos", "comite", "cientifico", "avaliacao", "resumo"),
    ("palestrante", "convite", "transporte", "diaria", "hospedagem", "passagem", "externo", "docente"),
    ("inscricao", "inscricoes", "taxa", "gratuita", "vagas", "credenciamento", "formulario", "participante"),
    ("patrocinio", "apoio", "empresa", "parceria", "divulgacao", "marca", "banner", "contrato"),
    ("seguranca", "portaria", "acesso", "horario", "noturno", "autorizacao", "vigilancia", "identificacao"),
)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_pgvector_enabled: bool = False


def compute_deterministic_embedding(
    text_content: str,
    dimensions: int = 16,
) -> list[float]:
    """Gera um vetor de embedding normalizado (L2) combinando conceitos de dominio e hashing lexical."""
    tokens = re.findall(r"[a-z0-9]+", (text_content or "").lower())
    vec = [0.0] * dimensions

    for idx, keywords in enumerate(_DOMAIN_CONCEPTS):
        if idx >= dimensions:
            break
        for token in tokens:
            if any(token.startswith(kw) or kw.startswith(token) for kw in keywords if len(token) >= 3):
                vec[idx] += 2.0

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = len(_DOMAIN_CONCEPTS) + (digest[0] % max(1, dimensions - len(_DOMAIN_CONCEPTS)))
        if bucket < dimensions:
            vec[bucket] += 1.0

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        vec[0] = 1.0
        return vec
    return [round(v / norm, 6) for v in vec]


def format_vector_literal(embedding: list[float]) -> str:
    """Formata uma lista de floats na sintaxe literal '[v1,v2,...]' exigida pelo tipo vector do pgvector."""
    return "[" + ",".join(f"{float(x):.6f}" for x in embedding) + "]"


def get_database_url() -> str:
    """Retorna a URL de conexao definida no ambiente ou o padrao local PostgreSQL."""
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def is_pgvector_enabled() -> bool:
    """Indica se a extensao vector (pgvector) esta ativa na conexao PostgreSQL atual."""
    return _pgvector_enabled


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
    global _engine, _session_factory, _pgvector_enabled
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    _pgvector_enabled = False


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


def _enable_pgvector_if_available(engine: Engine) -> None:
    """Ativa a extensao vector no PostgreSQL quando disponivel (imagem pgvector/pgvector:pg16)."""
    global _pgvector_enabled
    if engine.dialect.name != "postgresql":
        _pgvector_enabled = False
        return

    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        _pgvector_enabled = True
    except Exception:
        _pgvector_enabled = False


def run_migrations(revision: str = "head") -> None:
    """Executa migracoes versionadas do Alembic (upgrade para revision ou downgrade para 'base')."""
    from pathlib import Path
    from alembic import command
    from alembic.config import Config

    project_root = Path(__file__).resolve().parents[1]
    alembic_cfg = Config(str(project_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(project_root / "migrations"))
    alembic_cfg.set_main_option("sqlalchemy.url", get_database_url())

    engine = get_engine()
    with engine.begin() as conn:
        alembic_cfg.attributes["connection"] = conn
        if revision == "base":
            command.downgrade(alembic_cfg, "base")
        else:
            command.upgrade(alembic_cfg, revision)


def init_db() -> None:
    """Cria as tabelas de forma idempotente, ativa pgvector e popula usuarios e editais iniciais."""
    engine = get_engine()
    _enable_pgvector_if_available(engine)
    Base.metadata.create_all(bind=engine)
    _ensure_multi_session_columns(engine)

    with get_db_session() as session:
        existing_names = set(session.scalars(select(UserModel.name)).all())
        for name in DEFAULT_USERS:
            if name not in existing_names:
                session.add(UserModel(name=name))

        existing_titles = set(session.scalars(select(RegulationChunkModel.title)).all())
        for source, title, content in DEFAULT_REGULATIONS:
            if title not in existing_titles:
                vec = compute_deterministic_embedding(f"{title} {content}")
                session.add(
                    RegulationChunkModel(
                        source=source,
                        title=title,
                        content=content,
                        dimensions=len(vec),
                        embedding=vec,
                        embedding_text=format_vector_literal(vec),
                    )
                )
