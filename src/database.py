"""
Configuracao de conexao, gerenciamento de sessoes, migracoes Alembic e embeddings do banco de dados.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import functools
import hashlib
import logging
import math
import os
from pathlib import Path
import re
import unicodedata

from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from httpx import ConnectError, HTTPError
from langchain_ollama import OllamaEmbeddings
from ollama import RequestError, ResponseError
from sqlalchemy import Engine, create_engine, inspect, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.models import (
    Base,
    EMBEDDING_DIM,
    RegulationChunkModel,
    SessionModel,
    UserModel,
    format_vector_literal,
    set_pgvector_enabled,
)

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@localhost:5432/academic_flow"
)
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
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

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_pgvector_enabled: bool = False


def normalize_l2(vec: Sequence[float]) -> list[float]:
    """Aplica normalizacao L2 em um vetor numerico sem alterar sua dimensionalidade."""
    values = [float(x) for x in vec]
    if not values:
        raise ValueError("O vetor de embedding nao pode ser vazio.")
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        values[0] = 1.0
        return values
    return [round(v / norm, 6) for v in values]


def fit_embedding_dimensions(
    embedding: Sequence[float],
    dimensions: int = EMBEDDING_DIM,
) -> list[float]:
    """Valida a dimensao do vetor (permitindo zero-padding para vetores curtos de teste, sem dobrar dimensoes)."""
    if len(embedding) > dimensions:
        raise ValueError(
            f"Dimensao do vetor ({len(embedding)}) excede o tamanho da coluna vector({dimensions})."
        )
    padded = [float(x) for x in embedding] + [0.0] * (dimensions - len(embedding))
    return normalize_l2(padded)


def _normalize_text(text_content: str) -> str:
    """Remove acentos e normaliza o texto para minusculas."""
    normalized = unicodedata.normalize("NFKD", text_content or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()


def compute_trigram_embedding(
    text_content: str,
    dimensions: int = EMBEDDING_DIM,
) -> list[float]:
    """Fallback offline (para testes/CI sem Ollama) via projecao aleatoria de n-gramas."""
    cleaned = _normalize_text(text_content)
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    vec = [0.0] * dimensions

    features: list[tuple[str, float]] = []
    for token in tokens:
        features.append((f"w:{token}", 2.0))
        if len(token) >= 4:
            features.append((f"p:{token[:4]}", 1.5))
        if len(token) >= 3:
            padded = f"^{token}$"
            for i in range(len(padded) - 2):
                features.append((f"g:{padded[i : i + 3]}", 1.0))

    for feat, weight in features:
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[bucket] += sign * weight

    return normalize_l2(vec)


@functools.lru_cache(maxsize=4)
def _get_ollama_embedder(model_name: str, base_url: str) -> OllamaEmbeddings:
    """Retorna uma instancia cacheada de OllamaEmbeddings para evitar recriacao a cada consulta."""
    return OllamaEmbeddings(model=model_name, base_url=base_url)


def compute_ollama_embedding(
    text_content: str,
    dimensions: int = EMBEDDING_DIM,
) -> list[float]:
    """Gera embedding denso real via OllamaEmbeddings e valida estritamente a dimensao retornada."""
    model_name = (
        os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        or DEFAULT_EMBEDDING_MODEL
    )
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedder = _get_ollama_embedder(model_name, base_url)
    raw_vector = embedder.embed_query(text_content or "")
    if len(raw_vector) != dimensions:
        raise ValueError(
            f"O modelo '{model_name}' retornou {len(raw_vector)} dimensoes, "
            f"mas a coluna do banco exige exatamente {dimensions} dimensoes."
        )
    return normalize_l2(raw_vector)


def compute_embedding(
    text_content: str,
    dimensions: int = EMBEDDING_DIM,
) -> list[float]:
    """Gera o vetor de embedding usando Ollama em producao ou fallback de n-gramas quando offline."""
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
    if provider in {"trigram", "offline", "deterministic"}:
        return compute_trigram_embedding(text_content, dimensions=dimensions)

    try:
        return compute_ollama_embedding(text_content, dimensions=dimensions)
    except (
        ConnectError,
        HTTPError,
        ConnectionError,
        OSError,
        RequestError,
        ResponseError,
    ) as exc:
        logger.warning(
            "Falha ao gerar embedding via Ollama (%s); utilizando fallback de n-gramas.",
            exc,
        )
        return compute_trigram_embedding(text_content, dimensions=dimensions)


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
    set_pgvector_enabled(True)
    _get_ollama_embedder.cache_clear()


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


def _enable_pgvector_if_available(engine: Engine) -> None:
    """Ativa a extensao vector no PostgreSQL quando disponivel (imagem pgvector/pgvector:pg16)."""
    global _pgvector_enabled
    if engine.dialect.name != "postgresql":
        _pgvector_enabled = False
        set_pgvector_enabled(False)
        return

    try:
        with engine.begin() as conn:
            has_vector = conn.execute(
                text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
            ).scalar()
            if not has_vector:
                logger.warning(
                    "Extensao pgvector indisponivel no PostgreSQL; busca vetorial usara fallback em memoria."
                )
                _pgvector_enabled = False
                set_pgvector_enabled(False)
                return
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        _pgvector_enabled = True
        set_pgvector_enabled(True)
    except SQLAlchemyError as exc:
        logger.warning(
            "Extensao pgvector indisponivel no PostgreSQL (%s); busca vetorial usara fallback em memoria.",
            exc,
        )
        _pgvector_enabled = False
        set_pgvector_enabled(False)


def run_migrations(revision: str = "head") -> None:
    """Executa migracoes versionadas do Alembic sem apagar tabelas pre-existentes."""
    project_root = Path(__file__).resolve().parents[1]
    alembic_cfg = Config(str(project_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(project_root / "migrations"))
    alembic_cfg.set_main_option("sqlalchemy.url", get_database_url())
    alembic_cfg.attributes["configure_logger"] = False

    engine = get_engine()
    with engine.begin() as conn:
        alembic_cfg.attributes["connection"] = conn
        if revision == "base":
            command.downgrade(alembic_cfg, "base")
            return

        existing_tables = set(inspect(conn).get_table_names())
        if "users" in existing_tables and "alembic_version" not in existing_tables:
            logger.warning(
                "Banco pre-existente detectado sem tabela alembic_version; aplicando alembic stamp '%s' sem apagar dados.",
                revision,
            )
            Base.metadata.create_all(bind=conn)
            command.stamp(alembic_cfg, revision)
        else:
            command.upgrade(alembic_cfg, revision)


def _seed_initial_data() -> None:
    """Popula de forma idempotente a sessao padrao, os participantes e as normas institucionais iniciais."""
    with get_db_session() as session:
        if session.get(SessionModel, "default") is None:
            session.add(SessionModel(id="default"))

        existing_names = set(session.scalars(select(UserModel.name)).all())
        for name in DEFAULT_USERS:
            if name not in existing_names:
                session.add(UserModel(name=name))

        existing_titles = set(session.scalars(select(RegulationChunkModel.title)).all())
        for source, title, content in DEFAULT_REGULATIONS:
            if title not in existing_titles:
                vec = compute_embedding(f"{title} {content}")
                session.add(
                    RegulationChunkModel(
                        source=source,
                        title=title,
                        content=content,
                        embedding=vec,
                    )
                )


def init_db() -> None:
    """Inicializa o banco aplicando as migracoes do Alembic (DDL), ativando pgvector e executando o seed inicial."""
    engine = get_engine()
    _enable_pgvector_if_available(engine)
    run_migrations("head")
    _seed_initial_data()
