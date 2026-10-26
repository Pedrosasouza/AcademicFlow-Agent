"""
Camada de repositorio (Data Access Layer) centralizando todas as operacoes no banco de dados.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
import math
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from src import database
from src.database import (
    compute_embedding as _db_compute_embedding,
    fit_embedding_dimensions,
    format_vector_literal,
    get_db_session,
    get_engine,
)
from src.models import (
    ChatMessageModel,
    DecisionModel,
    DocumentSectionModel,
    RegulationChunkModel,
    TaskModel,
    TaskStatus,
    UserModel,
)

MAX_CONTEXT_MESSAGES = 20
_LEGACY_USER_PREFIX_RE = re.compile(r"^\[user:([^\]]+)\]\s*(.*)$", re.DOTALL)
_ISO_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _now() -> str:
    """Retorna timestamp compacto no formato esperado pelas ferramentas do agente."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _resolve_user(session: Session, raw_name: str | None) -> tuple[str, int | None]:
    """Resolve o nome normalizado e o ID relacional (FK) caso coincida com um usuario cadastrado."""
    cleaned = (raw_name or "").strip()
    if not cleaned:
        return "a definir", None

    users = session.scalars(select(UserModel)).all()
    for user in users:
        if user.name.lower() == cleaned.lower():
            return user.name, user.id
    return cleaned, None


def validate_task_status(status: str | TaskStatus) -> str:
    """Valida se o status informado pertence ao enum TaskStatus."""
    if isinstance(status, TaskStatus):
        return status.value
    cleaned = (status or "").strip().lower()
    try:
        return TaskStatus(cleaned).value
    except ValueError as exc:
        valid_values = ", ".join(s.value for s in TaskStatus)
        raise ValueError(
            f"Status invalido '{status}'. Valores permitidos: {valid_values}."
        ) from exc


def normalize_deadline(deadline: str | datetime | date | None) -> str:
    """Normaliza e valida um prazo (aceitando datetime/date, string ISO-8601 valida ou prazo textual)."""
    if deadline is None:
        return "a definir"
    if isinstance(deadline, datetime):
        return deadline.strftime("%Y-%m-%d %H:%M") if (deadline.hour or deadline.minute) else deadline.strftime("%Y-%m-%d")
    if isinstance(deadline, date):
        return deadline.isoformat()

    cleaned = str(deadline).strip()
    if not cleaned:
        return "a definir"

    if _ISO_DATE_PREFIX_RE.match(cleaned):
        try:
            parsed = datetime.fromisoformat(cleaned)
            return (
                parsed.strftime("%Y-%m-%d %H:%M")
                if ("T" in cleaned or " " in cleaned)
                else parsed.strftime("%Y-%m-%d")
            )
        except ValueError as exc:
            raise ValueError(
                f"Prazo em formato ISO invalido: '{cleaned}'. Use YYYY-MM-DD."
            ) from exc

    return cleaned


def _task_to_dict(task: TaskModel) -> dict[str, Any]:
    """Converte uma entidade TaskModel em dicionario serializavel."""
    return {
        "id": task.id,
        "title": task.title,
        "owner": task.owner,
        "owner_id": task.owner_id,
        "deadline": task.deadline,
        "status": task.status,
        "created_at": task.created_at,
    }


def _decision_to_dict(item: DecisionModel) -> dict[str, Any]:
    """Converte uma entidade DecisionModel em dicionario serializavel."""
    return {
        "id": item.id,
        "decision": item.decision,
        "reason": item.reason,
        "created_at": item.created_at,
    }


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Calcula a similaridade de cosseno exata entre dois vetores de mesma dimensao."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_embedding(text_content: str) -> list[float]:
    """Gera o vetor de embedding para indexacao e busca semantica nas normas da universidade."""
    return _db_compute_embedding(text_content)


def index_regulation_chunk(
    source: str,
    title: str,
    content: str,
    embedding: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Indexa um trecho de edital ou regulamento da universidade na coluna vetorial."""
    vec = (
        fit_embedding_dimensions(embedding)
        if embedding is not None
        else compute_embedding(f"{title} {content}")
    )
    with get_db_session() as session:
        chunk = RegulationChunkModel(
            source=source.strip() or "Norma Institucional",
            title=title.strip() or "Trecho sem titulo",
            content=content.strip(),
            embedding=vec,
        )
        session.add(chunk)
        session.flush()
        return {
            "id": chunk.id,
            "source": chunk.source,
            "title": chunk.title,
            "content": chunk.content,
            "dimensions": len(vec),
        }


def list_regulation_chunks() -> list[dict[str, Any]]:
    """Lista todos os trechos de editais e regulamentos cadastrados na base vetorial."""
    with get_db_session() as session:
        rows = session.scalars(
            select(RegulationChunkModel).order_by(RegulationChunkModel.id)
        ).all()
        return [
            {
                "id": row.id,
                "source": row.source,
                "title": row.title,
                "content": row.content,
            }
            for row in rows
        ]


def search_regulation_chunks(
    query: str,
    query_embedding: Sequence[float] | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Busca os trechos mais relevantes usando a coluna nativa vector(N) com indice HNSW no pgvector ou cosseno em memoria."""
    vec = (
        fit_embedding_dimensions(query_embedding)
        if query_embedding is not None
        else compute_embedding(query)
    )
    engine = get_engine()

    with get_db_session() as session:
        if engine.dialect.name == "postgresql" and database.is_pgvector_enabled():
            stmt = text(
                "SELECT id, source, title, content, "
                "1.0 - (embedding <=> CAST(:query_vec AS vector)) AS score "
                "FROM regulation_chunks "
                "ORDER BY embedding <=> CAST(:query_vec AS vector) "
                "LIMIT :top_k"
            )
            rows = session.execute(
                stmt,
                {
                    "query_vec": format_vector_literal(vec),
                    "top_k": top_k,
                },
            ).all()
            return [
                {
                    "id": row._mapping["id"],
                    "source": row._mapping["source"],
                    "title": row._mapping["title"],
                    "content": row._mapping["content"],
                    "score": float(row._mapping["score"]),
                }
                for row in rows
            ]

        candidates = session.scalars(select(RegulationChunkModel)).all()
        scored = [
            {
                "id": item.id,
                "source": item.source,
                "title": item.title,
                "content": item.content,
                "score": _cosine_similarity(vec, fit_embedding_dimensions(item.embedding)),
            }
            for item in candidates
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]


def list_users() -> list[str]:
    """Retorna a lista ordenada de participantes disponiveis no projeto."""
    with get_db_session() as session:
        names = session.scalars(select(UserModel.name).order_by(UserModel.id)).all()
        return list(names)


def list_sessions() -> list[str]:
    """Retorna a lista de identificadores de sessoes/eventos ativos no banco (incluindo 'default')."""
    with get_db_session() as session:
        found: set[str] = {"default"}
        for model in (TaskModel, DecisionModel, DocumentSectionModel, ChatMessageModel):
            values = session.scalars(select(model.session_id).distinct()).all()
            found.update(v for v in values if v)

        others = sorted(s for s in found if s != "default")
        return ["default", *others]


def create_task(
    title: str,
    owner: str | None = None,
    deadline: str | datetime | date | None = None,
    session_id: str = "default",
) -> dict[str, Any]:
    """Cria uma nova tarefa no banco para a sessao informada e retorna seus dados como dicionario."""
    normalized_deadline = normalize_deadline(deadline)
    with get_db_session() as session:
        normalized_owner, owner_id = _resolve_user(session, owner)

        task = TaskModel(
            session_id=session_id,
            title=title,
            owner=normalized_owner,
            owner_id=owner_id,
            deadline=normalized_deadline,
            status=TaskStatus.PENDING.value,
            created_at=_now(),
        )
        session.add(task)
        session.flush()
        return _task_to_dict(task)


def update_task_by_id(
    task_id: int,
    status: str | TaskStatus | None = None,
    owner: str | None = None,
    deadline: str | datetime | date | None = None,
    session_id: str = "default",
) -> dict[str, Any] | None:
    """Atualiza campos de uma tarefa existente na sessao pelo ID ou retorna None se nao encontrada."""
    validated_status = validate_task_status(status) if status else None
    normalized_deadline = normalize_deadline(deadline) if deadline else None

    with get_db_session() as session:
        task = session.get(TaskModel, task_id)
        if task is None or task.session_id != session_id:
            return None

        if validated_status:
            task.status = validated_status
        if owner:
            normalized_owner, owner_id = _resolve_user(session, owner)
            task.owner = normalized_owner
            task.owner_id = owner_id
        if normalized_deadline:
            task.deadline = normalized_deadline

        session.flush()
        return _task_to_dict(task)


def list_tasks(session_id: str = "default") -> list[dict[str, Any]]:
    """Lista todas as tarefas cadastradas na sessao ordenadas por ID."""
    with get_db_session() as session:
        tasks = session.scalars(
            select(TaskModel)
            .where(TaskModel.session_id == session_id)
            .order_by(TaskModel.id)
        ).all()
        return [_task_to_dict(task) for task in tasks]


def create_decision(
    decision: str,
    reason: str = "",
    session_id: str = "default",
) -> dict[str, Any]:
    """Registra uma nova decisao do grupo na sessao e retorna um dicionario."""
    with get_db_session() as session:
        item = DecisionModel(
            session_id=session_id,
            decision=decision,
            reason=reason or "",
            created_at=_now(),
        )
        session.add(item)
        session.flush()
        return _decision_to_dict(item)


def list_decisions(session_id: str = "default") -> list[dict[str, Any]]:
    """Lista todas as decisoes registradas na sessao ordenadas por ID."""
    with get_db_session() as session:
        decisions = session.scalars(
            select(DecisionModel)
            .where(DecisionModel.session_id == session_id)
            .order_by(DecisionModel.id)
        ).all()
        return [_decision_to_dict(item) for item in decisions]


def upsert_document_section(
    section: str,
    text_content: str,
    session_id: str = "default",
) -> dict[str, Any]:
    """Cria ou atualiza uma secao do documento colaborativo na sessao sem duplicar registros."""
    now_str = _now()
    with get_db_session() as session:
        existing = session.scalar(
            select(DocumentSectionModel).where(
                DocumentSectionModel.session_id == session_id,
                DocumentSectionModel.section == section,
            )
        )
        if existing is None:
            existing = DocumentSectionModel(
                session_id=session_id,
                section=section,
                text=text_content,
                updated_at=now_str,
            )
            session.add(existing)
        else:
            existing.text = text_content
            existing.updated_at = now_str

        session.flush()
        return {
            "section": existing.section,
            "text": existing.text,
            "updated_at": existing.updated_at,
        }


def get_document_sections(session_id: str = "default") -> dict[str, dict[str, str]]:
    """Retorna todas as secoes do documento da sessao no formato {nome_secao: {'text': ..., 'updated_at': ...}}."""
    with get_db_session() as session:
        rows = session.scalars(
            select(DocumentSectionModel)
            .where(DocumentSectionModel.session_id == session_id)
            .order_by(DocumentSectionModel.id)
        ).all()
        return {
            row.section: {
                "text": row.text,
                "updated_at": row.updated_at,
            }
            for row in rows
        }


def get_project_board(session_id: str = "default") -> dict[str, Any]:
    """Retorna o estado consolidado do quadro do projeto para a sessao informada."""
    return {
        "tasks": list_tasks(session_id=session_id),
        "decisions": list_decisions(session_id=session_id),
        "document_sections": get_document_sections(session_id=session_id),
    }


def _extract_author_and_clean_content(
    msg: HumanMessage,
    default_author: str | None = None,
) -> tuple[str | None, str]:
    """Extrai o autor da mensagem e remove eventual prefixo legado '[user:X]' do conteudo."""
    raw_content = str(msg.content or "")
    explicit_author = (
        getattr(msg, "name", None)
        or msg.additional_kwargs.get("author")
        or default_author
    )

    match = _LEGACY_USER_PREFIX_RE.match(raw_content)
    if match:
        prefix_author, clean_body = match.group(1).strip(), match.group(2)
        return (explicit_author or prefix_author or None), clean_body

    cleaned_author = str(explicit_author).strip() if explicit_author else None
    return (cleaned_author or None), raw_content


def append_messages(
    messages: Sequence[BaseMessage],
    session_id: str = "default",
    default_author: str | None = None,
) -> None:
    """Persiste apenas as novas mensagens de um turno mantendo a ordem cronologica e autoria relacional."""
    if not messages:
        return

    with get_db_session() as session:
        for msg in messages:
            if isinstance(msg, HumanMessage):
                raw_author, clean_content = _extract_author_and_clean_content(
                    msg, default_author=default_author
                )
                author_name: str | None = None
                user_id: int | None = None
                if raw_author:
                    resolved_name, resolved_id = _resolve_user(session, raw_author)
                    author_name = resolved_name
                    user_id = resolved_id

                session.add(
                    ChatMessageModel(
                        session_id=session_id,
                        role="human",
                        author=author_name,
                        user_id=user_id,
                        content=clean_content,
                        tool_calls=None,
                        tool_call_id=None,
                    )
                )
            elif isinstance(msg, AIMessage):
                raw_tool_calls = getattr(msg, "tool_calls", None)
                serialized_calls = list(raw_tool_calls) if raw_tool_calls else None
                session.add(
                    ChatMessageModel(
                        session_id=session_id,
                        role="ai",
                        author=None,
                        user_id=None,
                        content=str(msg.content or ""),
                        tool_calls=serialized_calls,
                        tool_call_id=None,
                    )
                )
            elif isinstance(msg, ToolMessage):
                session.add(
                    ChatMessageModel(
                        session_id=session_id,
                        role="tool",
                        author=None,
                        user_id=None,
                        content=str(msg.content),
                        tool_calls=None,
                        tool_call_id=getattr(msg, "tool_call_id", None),
                    )
                )


def load_messages(session_id: str = "default") -> list[BaseMessage]:
    """Carrega o historico completo de mensagens de uma sessao convertido para classes do LangChain."""
    with get_db_session() as session:
        rows = session.scalars(
            select(ChatMessageModel)
            .where(ChatMessageModel.session_id == session_id)
            .order_by(ChatMessageModel.id)
        ).all()

        history: list[BaseMessage] = []
        for row in rows:
            if row.role == "human":
                if row.author:
                    history.append(HumanMessage(content=row.content, name=row.author))
                else:
                    history.append(HumanMessage(content=row.content))
            elif row.role == "ai":
                history.append(
                    AIMessage(
                        content=row.content,
                        tool_calls=row.tool_calls or [],
                    )
                )
            elif row.role == "tool":
                history.append(
                    ToolMessage(
                        content=row.content,
                        tool_call_id=row.tool_call_id or "unknown_call",
                    )
                )
        return history


def load_context_messages(
    session_id: str = "default",
    max_messages: int = MAX_CONTEXT_MESSAGES,
) -> list[BaseMessage]:
    """Retorna as ultimas mensagens limitadas para o contexto do LLM sem deixar ToolMessage orfa no inicio."""
    all_messages = load_messages(session_id=session_id)
    if len(all_messages) <= max_messages:
        return all_messages

    sliced = all_messages[-max_messages:]
    while sliced and isinstance(sliced[0], ToolMessage):
        sliced.pop(0)
    return sliced


def clear_project_board(session_id: str | None = None) -> None:
    """Remove tarefas, decisoes, secoes e mensagens (de uma sessao especifica ou de todo o banco)."""
    engine = get_engine()
    with get_db_session() as session:
        if session_id is not None:
            session.execute(delete(TaskModel).where(TaskModel.session_id == session_id))
            session.execute(
                delete(DecisionModel).where(DecisionModel.session_id == session_id)
            )
            session.execute(
                delete(DocumentSectionModel).where(
                    DocumentSectionModel.session_id == session_id
                )
            )
            session.execute(
                delete(ChatMessageModel).where(ChatMessageModel.session_id == session_id)
            )
        elif engine.dialect.name == "postgresql":
            session.execute(
                text(
                    "TRUNCATE TABLE tasks, decisions, document_sections, chat_messages "
                    "RESTART IDENTITY CASCADE"
                )
            )
        else:
            session.execute(delete(TaskModel))
            session.execute(delete(DecisionModel))
            session.execute(delete(DocumentSectionModel))
            session.execute(delete(ChatMessageModel))
