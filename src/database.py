"""
Configuracao de conexao, gerenciamento de sessoes, migracoes Alembic e embeddings do banco de dados.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import re
import unicodedata

from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from sqlalchemy import Engine, create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.models import Base, EMBEDDING_DIM, RegulationChunkModel, UserModel

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

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_pgvector_enabled: bool = False


def fit_embedding_dimensions(
    embedding: Sequence[float],
    dimensions: int = EMBEDDING_DIM,
) -> list[float]:
    """Ajusta qualquer vetor para a dimensao fixa do banco e aplica normalizacao L2."""
    vec = [0.0] * dimensions
    for idx, val in enumerate(embedding):
        vec[idx % dimensions] += float(val)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        vec[0] = 1.0
        return vec
    return [round(v / norm, 6) for v in vec]


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

    return fit_embedding_dimensions(vec, dimensions=dimensions)


def compute_ollama_embedding(
    text_content: str,
    dimensions: int = EMBEDDING_DIM,
) -> list[float]:
    """Gera embedding denso real via OllamaEmbeddings e ajusta para a dimensao da coluna vector."""
    model_name = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        os.getenv("OLLAMA_MODEL", "nomic-embed-text"),
    )
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedder = OllamaEmbeddings(model=model_name, base_url=base_url)
    raw_vector = embedder.embed_query(text_content or "")
    return fit_embedding_dimensions(raw_vector, dimensions=dimensions)


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
    except Exception:
        return compute_trigram_embedding(text_content, dimensions=dimensions)


def format_vector_literal(embedding: Sequence[float]) -> str:
    """Formata uma sequencia de floats na sintaxe literal '[v1,v2,...]' exigida pelo pgvector."""
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
    """Executa migracoes versionadas do Alembic como fonte unica da verdade para o DDL."""
    project_root = Path(__file__).resolve().parents[1]
    alembic_cfg = Config(str(project_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(project_root / "migrations"))
    alembic_cfg.set_main_option("sqlalchemy.url", get_database_url())
    alembic_cfg.attributes["configure_logger"] = False

    engine = get_engine()
    with engine.begin() as conn:
        if revision != "base":
            existing_tables = set(inspect(conn).get_table_names())
            if "users" in existing_tables and "alembic_version" not in existing_tables:
                Base.metadata.drop_all(bind=conn)

        alembic_cfg.attributes["connection"] = conn
        if revision == "base":
            command.downgrade(alembic_cfg, "base")
        else:
            command.upgrade(alembic_cfg, revision)


def _seed_initial_data() -> None:
    """Popula de forma idempotente os participantes e normas institucionais iniciais."""
    with get_db_session() as session:
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
    """Inicializa o banco aplicando as migracoes do Alembic (DDL) e o seed inicial (DML)."""
    engine = get_engine()
    _enable_pgvector_if_available(engine)
    run_migrations("head")
    _seed_initial_data()
