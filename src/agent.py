"""
Sistema colaborativo com ReAct Agent usando LangGraph e persistencia relacional.

Cenario escolhido:
    Planejamento colaborativo de um evento academico.

O agente ajuda um grupo a conversar, registrar tarefas, registrar decisoes
e consultar o quadro atual do planejamento. Os dados sao persistidos em banco
de dados via SQLAlchemy 2.0 (PostgreSQL em producao/local e SQLite em testes).
"""

from __future__ import annotations

from contextvars import ContextVar
import os
from typing import Any

from dotenv import load_dotenv
from httpx import ConnectError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy.exc import OperationalError

from src import repository
from src.database import init_db

load_dotenv()


# =============================================================================
# 1. CONTEXTO DE SESSAO E ADAPTADOR DE LEITURA DO QUADRO DO PROJETO
# =============================================================================

_active_session_var: ContextVar[str] = ContextVar("active_session_id", default="default")


def get_active_session() -> str:
    """Retorna o identificador da sessao/evento ativo no contexto de execucao atual."""
    return _active_session_var.get()


def set_active_session(session_id: str) -> None:
    """Define o identificador da sessao/evento ativo para as ferramentas e leituras do quadro."""
    cleaned = (session_id or "").strip() or "default"
    _active_session_var.set(cleaned)


class ProjectBoardView:
    """Adaptador de leitura que consulta sempre o banco de dados na sessao ativa."""

    def to_dict(self) -> dict[str, Any]:
        """Retorna um snapshot atualizado do quadro do projeto a partir do repositorio."""
        return repository.get_project_board(session_id=get_active_session())

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self.to_dict() == other
        if isinstance(other, ProjectBoardView):
            return self.to_dict() == other.to_dict()
        return False


project_board = ProjectBoardView()


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
    task = repository.create_task(
        title=title,
        owner=owner,
        deadline=deadline,
        session_id=get_active_session(),
    )
    return (
        f"Tarefa #{task['id']} registrada: {task['title']} | "
        f"responsavel: {task['owner']} | prazo: {task['deadline']}."
    )


@tool
def update_task(
    task_id: int,
    status: str | None = None,
    owner: str | None = None,
    deadline: str | None = None,
) -> str:
    """Update an existing planning task's status, owner, or deadline.

    Use this when a participant asks to complete, reassign, reschedule, or update a task.
    """
    updated = repository.update_task_by_id(
        task_id=task_id,
        status=status,
        owner=owner,
        deadline=deadline,
        session_id=get_active_session(),
    )
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


tools = [
    register_task,
    update_task,
    register_decision,
    update_document_section,
    consult_project_board,
]


# =============================================================================
# 3. LLM + PROMPT DO SISTEMA
# =============================================================================

SYSTEM_PROMPT = """
Voce e um agente colaborativo para planejamento de um evento academico.

Seu papel:
- facilitar a conversa entre participantes;
- transformar combinados em tarefas, decisoes e trechos de documento;
- consultar o quadro do projeto quando alguem pedir status;
- responder diretamente quando a pergunta for apenas conversa ou orientacao.

Use ferramentas quando o usuario pedir para registrar, atualizar, consultar
ou organizar informacoes do projeto. Depois de usar uma ferramenta, explique
em linguagem natural o que foi feito e sugira o proximo passo colaborativo.
""".strip()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
).bind_tools(tools)


# =============================================================================
# 4. GRAFO REACT COM LANGGRAPH
# =============================================================================


def agent(state: MessagesState) -> MessagesState:
    """Call the LLM. It may answer directly or request tool calls."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

react_event_agent = graph.compile()


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

        context_history = repository.load_context_messages(session_id=active_session)
        user_message = HumanMessage(content=user_input)
        pending_history = [*context_history, user_message]
        initial_count = len(context_history)
        step = None

        try:
            for step in react_event_agent.stream(
                {"messages": pending_history},
                stream_mode="values",
            ):
                if len(step["messages"]) > initial_count + 1:
                    print_step(step)
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

        if step is not None:
            new_messages = step["messages"][initial_count:]
            repository.append_messages(new_messages, session_id=active_session)
        print()


if __name__ == "__main__":
    main()
