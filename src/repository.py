"""
Camada de repositorio (Data Access Layer) centralizando todas as operacoes no banco de dados.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from src.database import get_db_session, get_engine, init_db
from src.models import (
    ChatMessageModel,
    DecisionModel,
    DocumentSectionModel,
    TaskModel,
    UserModel,
)

MAX_CONTEXT_MESSAGES = 20


def _now() -> str:
    """Retorna timestamp compacto no formato esperado pelas ferramentas do agente."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _normalize_owner(session: Session, owner: str | None) -> str:
    """Normaliza o nome do responsavel caso coincida (case-insensitive) com um usuario cadastrado."""
    cleaned = (owner or "").strip()
    if not cleaned:
        return "a definir"

    registered_users = session.scalars(select(UserModel.name)).all()
    for user_name in registered_users:
        if user_name.lower() == cleaned.lower():
            return user_name
    return cleaned


def _task_to_dict(task: TaskModel) -> dict[str, Any]:
    """Converte uma entidade TaskModel em dicionario serializavel."""
    return {
        "id": task.id,
        "title": task.title,
        "owner": task.owner,
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


def list_users() -> list[str]:
    """Retorna a lista ordenada de participantes disponiveis no projeto."""
    init_db()
    with get_db_session() as session:
        names = session.scalars(select(UserModel.name).order_by(UserModel.id)).all()
        return list(names)


def create_task(
    title: str,
    owner: str | None = None,
    deadline: str | None = None,
) -> dict[str, Any]:
    """Cria uma nova tarefa no banco e retorna seus dados como dicionario."""
    init_db()
    with get_db_session() as session:
        normalized_owner = _normalize_owner(session, owner)
        normalized_deadline = (deadline or "").strip() or "a definir"

        task = TaskModel(
            title=title,
            owner=normalized_owner,
            deadline=normalized_deadline,
            status="pendente",
            created_at=_now(),
        )
        session.add(task)
        session.flush()
        return _task_to_dict(task)


def update_task_by_id(
    task_id: int,
    status: str | None = None,
    owner: str | None = None,
    deadline: str | None = None,
) -> dict[str, Any] | None:
    """Atualiza campos de uma tarefa existente pelo ID ou retorna None se nao encontrada."""
    init_db()
    with get_db_session() as session:
        task = session.get(TaskModel, task_id)
        if task is None:
            return None

        if status:
            task.status = status
        if owner:
            task.owner = _normalize_owner(session, owner)
        if deadline:
            task.deadline = deadline

        session.flush()
        return _task_to_dict(task)


def list_tasks() -> list[dict[str, Any]]:
    """Lista todas as tarefas cadastradas ordenadas por ID."""
    init_db()
    with get_db_session() as session:
        tasks = session.scalars(select(TaskModel).order_by(TaskModel.id)).all()
        return [_task_to_dict(task) for task in tasks]


def create_decision(decision: str, reason: str = "") -> dict[str, Any]:
    """Registra uma nova decisao do grupo no banco e retorna um dicionario."""
    init_db()
    with get_db_session() as session:
        item = DecisionModel(
            decision=decision,
            reason=reason or "",
            created_at=_now(),
        )
        session.add(item)
        session.flush()
        return _decision_to_dict(item)


def list_decisions() -> list[dict[str, Any]]:
    """Lista todas as decisoes registradas ordenadas por ID."""
    init_db()
    with get_db_session() as session:
        decisions = session.scalars(
            select(DecisionModel).order_by(DecisionModel.id)
        ).all()
        return [_decision_to_dict(item) for item in decisions]


def upsert_document_section(section: str, text_content: str) -> dict[str, Any]:
    """Cria ou atualiza uma secao do documento colaborativo sem duplicar registros."""
    init_db()
    now_str = _now()
    with get_db_session() as session:
        existing = session.scalar(
            select(DocumentSectionModel).where(DocumentSectionModel.section == section)
        )
        if existing is None:
            existing = DocumentSectionModel(
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


def get_document_sections() -> dict[str, dict[str, str]]:
    """Retorna todas as secoes do documento no formato {nome_secao: {'text': ..., 'updated_at': ...}}."""
    init_db()
    with get_db_session() as session:
        rows = session.scalars(
            select(DocumentSectionModel).order_by(DocumentSectionModel.id)
        ).all()
        return {
            row.section: {
                "text": row.text,
                "updated_at": row.updated_at,
            }
            for row in rows
        }


def get_project_board() -> dict[str, Any]:
    """Retorna o estado consolidado do quadro do projeto a partir do banco de dados."""
    return {
        "tasks": list_tasks(),
        "decisions": list_decisions(),
        "document_sections": get_document_sections(),
    }


def append_messages(
    messages: Sequence[BaseMessage],
    session_id: str = "default",
) -> None:
    """Persiste apenas as novas mensagens de um turno mantendo a ordem cronologica."""
    if not messages:
        return

    init_db()
    with get_db_session() as session:
        for msg in messages:
            if isinstance(msg, HumanMessage):
                session.add(
                    ChatMessageModel(
                        session_id=session_id,
                        role="human",
                        content=str(msg.content),
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
                        content=str(msg.content),
                        tool_calls=None,
                        tool_call_id=getattr(msg, "tool_call_id", None),
                    )
                )


def load_messages(session_id: str = "default") -> list[BaseMessage]:
    """Carrega o historico completo de mensagens de uma sessao convertido para classes do LangChain."""
    init_db()
    with get_db_session() as session:
        rows = session.scalars(
            select(ChatMessageModel)
            .where(ChatMessageModel.session_id == session_id)
            .order_by(ChatMessageModel.id)
        ).all()

        history: list[BaseMessage] = []
        for row in rows:
            if row.role == "human":
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


def clear_project_board() -> None:
    """Remove tarefas, decisoes, secoes e mensagens do chat, preservando os usuarios cadastrados."""
    init_db()
    engine = get_engine()
    with get_db_session() as session:
        if engine.dialect.name == "postgresql":
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
