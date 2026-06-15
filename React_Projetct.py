"""
Draft de sistema colaborativo com React Agent usando LangGraph.

Cenario escolhido:
    Planejamento colaborativo de um evento academico.

O agente ajuda um grupo a conversar, registrar tarefas, registrar decisoes
e consultar o quadro atual do planejamento. O foco deste prototipo e mostrar
o fluxo React: o LLM decide se responde diretamente ou se chama uma ferramenta.

Prerequisitos:
    pip install langgraph langchain-ollama langchain-core
    ollama pull llama3.2
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from httpx import ConnectError
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition


# =============================================================================
# 1. MEMORIA SIMPLES DO PROJETO
# =============================================================================

project_board: dict[str, Any] = {
    "tasks": [],
    "decisions": [],
    "document_sections": {},
}

users = ["Alice", "Bob", "Charlie"]

def get_users():
    return users

def _now() -> str:
    """Return a compact timestamp for records created by tools."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


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
    owner = owner or "a definir"
    deadline = deadline or "a definir"

    task = {
        "id": len(project_board["tasks"]) + 1,
        "title": title,
        "owner": owner,
        "deadline": deadline,
        "status": "pendente",
        "created_at": _now(),
    }
    project_board["tasks"].append(task)
    return (
        f"Tarefa #{task['id']} registrada: {title} | "
        f"responsavel: {owner} | prazo: {deadline}."
    )


@tool
def register_decision(decision: str, reason: str = "") -> str:
    """Register an important group decision.

    Use this when participants agree on something that should be preserved.
    """
    item = {
        "id": len(project_board["decisions"]) + 1,
        "decision": decision,
        "reason": reason,
        "created_at": _now(),
    }
    project_board["decisions"].append(item)
    reason_text = f" Motivo: {reason}" if reason else ""
    return f"Decisao #{item['id']} registrada: {decision}.{reason_text}"


@tool
def update_document_section(section: str, text: str) -> str:
    """Create or update a collaborative document section.

    Use this when the group asks to draft, rewrite, or save part of the event plan.
    """
    project_board["document_sections"][section] = {
        "text": text,
        "updated_at": _now(),
    }
    return f"Secao '{section}' atualizada com {len(text.split())} palavras."


@tool
def consult_project_board() -> str:
    """Show the current tasks, decisions, and document sections.

    Use this when participants ask for a summary, status, or current project board.
    """
    tasks = project_board["tasks"]
    decisions = project_board["decisions"]
    sections = project_board["document_sections"]

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


llm = ChatOllama(model="llama3.2").bind_tools(tools)


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
# 5. LOOP DE CONVERSA
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
    print("=== React Agent Colaborativo: Planejamento de Evento ===")
    print("Digite 'exit' ou 'quit' para sair.\n")
    print("Antes de conversar, confirme que o Ollama esta aberto.")
    print("Comandos uteis: ollama serve | ollama pull llama3.2\n")

    history = []

    while True:
        user_input = input("Grupo: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        history.append(("user", user_input))

        try:
            for step in react_event_agent.stream(
                {"messages": history},
                stream_mode="values",
            ):
                print_step(step)
        except ConnectError:
            print("\nNao consegui conectar ao Ollama.")
            print("Abra outro terminal e execute: ollama serve")
            print("Se o modelo ainda nao existir, execute: ollama pull llama3.2")
            print("Depois rode este programa novamente.\n")
            break

        history = step["messages"]
        print()


if __name__ == "__main__":
    main()
