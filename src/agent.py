"""
Sistema colaborativo com ReAct Agent usando LangGraph e persistencia relacional.

Cenario escolhido:
    Planejamento colaborativo de um evento academico.

O agente ajuda um grupo a conversar, registrar tarefas, registrar decisoes
e consultar o quadro atual do planejamento. Os dados sao persistidos em banco
de dados via SQLAlchemy 2.0 (PostgreSQL em producao/local e SQLite em testes).
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
import logging
import os
from typing import Any

from dotenv import load_dotenv
from httpx import ConnectError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy.exc import OperationalError

from src import repository
from src.database import init_db
from src.models import TaskStatus

load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# 1. CONTEXTO DE SESSAO E PARTICIPANTE ATIVO
# =============================================================================

_active_session_var: ContextVar[str] = ContextVar("active_session_id", default="default")
_active_user_var: ContextVar[str | None] = ContextVar("active_user", default=None)
_SELF_REFERENCES: frozenset[str] = frozenset({"eu", "mim", "me", "meu", "minha"})


def get_active_session() -> str:
    """Retorna o identificador da sessao/evento ativo no contexto de execucao atual."""
    return _active_session_var.get()


def set_active_session(session_id: str) -> None:
    """Define o identificador da sessao/evento ativo para as ferramentas e leituras do quadro."""
    cleaned = (session_id or "").strip() or "default"
    _active_session_var.set(cleaned)


def get_active_user() -> str | None:
    """Retorna o nome do participante ativo no turno de conversa atual."""
    return _active_user_var.get()


def set_active_user(user: str | None) -> None:
    """Define o participante ativo no contexto para atribuicao automatica nas ferramentas."""
    cleaned = (user or "").strip() or None
    _active_user_var.set(cleaned)


def get_users() -> list[str]:
    """Retorna a lista de participantes disponiveis na sessao colaborativa."""
    return repository.list_users()


# =============================================================================
# 2. FERRAMENTAS DO AGENTE
# =============================================================================


@tool
def register_task(
    title: str,
    owner: str | None = None,
    deadline: str | None = None,
) -> str:
    """Register a planning task for the group.

    Use this when a participant asks to create, add, assign, or remember a task.
    """
    cleaned_owner = (owner or "").strip()
    if not cleaned_owner or cleaned_owner.lower() in _SELF_REFERENCES:
        resolved_owner = get_active_user()
    else:
        resolved_owner = cleaned_owner

    try:
        task = repository.create_task(
            title=title,
            owner=resolved_owner,
            deadline=deadline,
            session_id=get_active_session(),
        )
    except ValueError as exc:
        logger.warning("Erro de validacao em register_task: %s", exc)
        return f"Erro de validacao ao criar tarefa: {exc}"

    return (
        f"Tarefa #{task['id']} registrada: {task['title']} | "
        f"responsavel: {task['owner']} | prazo: {task['deadline']}."
    )


@tool
def update_task(
    task_id: int,
    status: TaskStatus | str | None = None,
    owner: str | None = None,
    deadline: str | None = None,
) -> str:
    """Update an existing planning task's status, owner, or deadline.

    Allowed status values: 'pendente', 'em_andamento', 'concluida', 'cancelada'.
    Use this when a participant asks to complete, reassign, reschedule, or update a task.
    """
    try:
        updated = repository.update_task_by_id(
            task_id=task_id,
            status=status,
            owner=owner,
            deadline=deadline,
            session_id=get_active_session(),
        )
    except ValueError as exc:
        return f"Erro de validacao ao atualizar tarefa #{task_id}: {exc}"

    if updated is None:
        return f"Tarefa #{task_id} nao encontrada."

    return (
        f"Tarefa #{updated['id']} atualizada: {updated['title']} | "
        f"responsavel: {updated['owner']} | prazo: {updated['deadline']} | "
        f"status: {updated['status']}."
    )


@tool
def register_decision(decision: str, reason: str = "") -> str:
    """Register an important group decision.

    Use this when participants agree on something that should be preserved.
    """
    item = repository.create_decision(
        decision=decision,
        reason=reason,
        session_id=get_active_session(),
    )
    reason_text = f" Motivo: {item['reason']}" if item["reason"] else ""
    return f"Decisao #{item['id']} registrada: {item['decision']}.{reason_text}"


@tool
def update_document_section(section: str, text: str) -> str:
    """Create or update a collaborative document section.

    Use this when the group asks to draft, rewrite, or save part of the event plan.
    """
    repository.upsert_document_section(
        section=section,
        text_content=text,
        session_id=get_active_session(),
    )
    return f"Secao '{section}' atualizada com {len(text.split())} palavras."


@tool
def consult_project_board() -> str:
    """Show the current tasks, decisions, and document sections.

    Use this when participants ask for a summary, status, or current project board.
    """
    board = repository.get_project_board(session_id=get_active_session())
    tasks = board["tasks"]
    decisions = board["decisions"]
    sections = board["document_sections"]

    task_lines = [
        f"- #{task['id']} {task['title']} | {task['owner']} | "
        f"{task['deadline']} | {task['status']}"
        for task in tasks
    ] or ["- nenhuma tarefa registrada"]

    decision_lines = [
        f"- #{item['id']} {item['decision']}"
        for item in decisions
    ] or ["- nenhuma decisao registrada"]

    section_lines = [
        f"- {name}: {len(data['text'].split())} palavras"
        for name, data in sections.items()
    ] or ["- nenhuma secao criada"]

    return "\n".join(
        [
            "Tarefas:",
            *task_lines,
            "",
            "Decisoes:",
            *decision_lines,
            "",
            "Secoes do documento:",
            *section_lines,
        ]
    )


@tool
def search_event_regulations(query: str) -> str:
    """Search university regulations, calls for proposals (editais), and academic event rules.

    Use this when participants ask about university rules, auditorium reservation deadlines,
    coffee break budget limits, certificate requirements, or paper submission guidelines.
    """
    matches = repository.search_regulation_chunks(query=query, top_k=3)
    if not matches:
        return "Nenhuma norma ou edital encontrado para a consulta."

    lines = ["Normas e editais encontrados:"]
    for item in matches:
        lines.append(
            f"- [{item['source']} | {item['title']}] {item['content']}"
        )
    return "\n".join(lines)


tools = [
    register_task,
    update_task,
    register_decision,
    update_document_section,
    consult_project_board,
    search_event_regulations,
]


# =============================================================================
# 3. LLM + PROMPT DO SISTEMA + BUILDER DO GRAFO REACT
# =============================================================================

SYSTEM_PROMPT = """
Voce e um agente colaborativo para planejamento de um evento academico.

Seu papel:
- facilitar a conversa entre participantes;
- transformar combinados em tarefas, decisoes e trechos de documento;
- consultar o quadro do projeto quando alguem pedir status;
- consultar normas e editais da universidade (search_event_regulations) quando houver duvidas sobre regras institucionais, prazos de reserva, orcamento ou certificados;
- responder diretamente quando a pergunta for apenas conversa ou orientacao.

Use ferramentas quando o usuario pedir para registrar, atualizar, consultar
ou organizar informacoes do projeto. Depois de usar uma ferramenta, explique
em linguagem natural o que foi feito e sugira o proximo passo colaborativo.
""".strip()


def build_llm(
    model: str | None = None,
    base_url: str | None = None,
) -> Any:
    """Cria uma instancia do modelo ChatOllama vinculada as ferramentas do agente."""
    resolved_model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
    resolved_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=resolved_model,
        base_url=resolved_url,
        temperature=0,
    ).bind_tools(tools)


llm = build_llm()


def _format_messages_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Injeta o nome do participante no texto enviado ao LLM sem poluir a mensagem persistida."""
    formatted: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in messages:
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None):
            text_content = str(msg.content or "")
            if not text_content.startswith(f"[{msg.name}]"):
                formatted.append(
                    HumanMessage(content=f"[{msg.name}] {text_content}", name=msg.name)
                )
                continue
        formatted.append(msg)
    return formatted


def build_agent(custom_llm: Any = None) -> Any:
    """Constroi e compila o StateGraph ReAct de forma isolada."""
    def _agent_node(state: MessagesState) -> MessagesState:
        active_llm = custom_llm if custom_llm is not None else llm
        messages = _format_messages_for_llm(state["messages"])
        response = active_llm.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


react_event_agent = build_agent()


# =============================================================================
# 4. SERVICO UNIFICADO DE TURNO (CLI & UI)
# =============================================================================


def run_turn(
    user_input: str,
    session_id: str = "default",
    author: str | None = None,
    on_step: Callable[[MessagesState], None] | None = None,
    compiled_agent: Any = None,
) -> list[BaseMessage]:
    """Executa um turno completo de conversa (contexto -> grafo ReAct -> persistencia no banco)."""
    cleaned_author = (author or "").strip() or None
    set_active_session(session_id)
    set_active_user(cleaned_author)
    context_history = repository.load_context_messages(session_id=session_id)
    user_message = (
        HumanMessage(content=user_input, name=cleaned_author)
        if cleaned_author
        else HumanMessage(content=user_input)
    )
    pending_history = [*context_history, user_message]
    initial_count = len(context_history)

    runner = compiled_agent if compiled_agent is not None else react_event_agent
    final_step: MessagesState | None = None

    for step in runner.stream(
        {"messages": pending_history},
        stream_mode="values",
    ):
        final_step = step
        if on_step is not None and len(step["messages"]) > initial_count + 1:
            on_step(step)

    if final_step is None:
        return []

    new_messages = list(final_step["messages"][initial_count:])
    repository.append_messages(
        new_messages,
        session_id=session_id,
        default_author=cleaned_author,
    )
    return new_messages


# =============================================================================
# 5. LOOP DE CONVERSA (CLI)
# =============================================================================


def print_step(step: MessagesState) -> None:
    """Print relevant events from each graph step."""
    msg = step["messages"][-1]
    role = getattr(msg, "type", "unknown")

    if role == "ai" and getattr(msg, "tool_calls", None):
        names = [tool_call["name"] for tool_call in msg.tool_calls]
        print(f"  [chamando ferramenta: {', '.join(names)}]")
    elif role == "tool":
        print(f"  [resultado da ferramenta: {msg.content}]")
    elif role == "ai" and msg.content:
        print(f"Agente: {msg.content}")


def main() -> None:
    """Run the interactive CLI loop for the collaborative event planning agent."""
    print("=== React Agent Colaborativo: Planejamento de Evento ===")
    print("Digite 'exit' ou 'quit' para sair.\n")
    print("Antes de conversar, confirme que o PostgreSQL e o Ollama estao ativos.")
    print("Comandos uteis: docker compose up -d | ollama serve | ollama pull llama3.2\n")

    try:
        init_db()
    except OperationalError:
        print("\nNao consegui conectar ao banco de dados PostgreSQL.")
        print("Inicie o container com: docker compose up -d")
        print("Ou verifique a variavel DATABASE_URL no seu arquivo .env.\n")
        return

    active_session = get_active_session()

    while True:
        user_input = input("Grupo: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        try:
            run_turn(
                user_input=user_input,
                session_id=active_session,
                on_step=print_step,
            )
        except ConnectError:
            print("\nNao consegui conectar ao Ollama.")
            print("Abra outro terminal e execute: ollama serve")
            print("Se o modelo ainda nao existir, execute: ollama pull llama3.2")
            print("Depois rode este programa novamente.\n")
            break
        except OperationalError:
            print("\nErro de conexao com o banco de dados durante a execucao.")
            print("Verifique se o container PostgreSQL continua ativo (docker compose ps).\n")
            break

        print()


if __name__ == "__main__":
    main()
