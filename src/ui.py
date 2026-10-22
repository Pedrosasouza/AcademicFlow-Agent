from pathlib import Path
import sys

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from httpx import ConnectError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import agent as react_agent

LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"


def _message_role(message) -> str | None:
    """Return the chat role ('user' or 'assistant') for displayable messages."""
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        if getattr(message, "tool_calls", None) or not message.content:
            return None
        return "assistant"
    return None


def _render_tools_called() -> None:
    """Render badges in the sidebar for each tool invoked during the session."""
    with st.sidebar:
        for tool_name in st.session_state.tools_called:
            st.badge(f"Ferramenta chamada: {tool_name}", color="orange")


def _reset_session() -> None:
    """Clear message history, called tools, and the shared project board."""
    react_agent.project_board["tasks"].clear()
    react_agent.project_board["decisions"].clear()
    react_agent.project_board["document_sections"].clear()
    st.session_state.messages = []
    st.session_state.tools_called = set()
    st.session_state.project_board = react_agent.project_board


def _render_user_chooser() -> None:
    """Render the participant selector and session reset button in the sidebar."""
    with st.sidebar:
        user = st.selectbox("Selecione o usuário", usuarios)
        st.session_state.current_user = user
        if st.button("Limpar sessão"):
            _reset_session()
            st.rerun()


def _render_board_summary() -> None:
    """Render the project board metrics, tasks, decisions, and document sections."""
    with st.sidebar:
        st.markdown("### Quadro do projeto")
        board = st.session_state.project_board
        cols = st.columns(3)
        cols[0].metric("Tarefas", len(board["tasks"]))
        cols[1].metric("Decisões", len(board["decisions"]))
        cols[2].metric("Seções", len(board["document_sections"]))

        with st.expander("Tarefas", expanded=True):
            if board["tasks"]:
                for task in board["tasks"]:
                    st.markdown(
                        f"**#{task['id']} {task['title']}**  \n"
                        f"Responsável: {task['owner']}  \n"
                        f"Prazo: {task['deadline']}  \n"
                        f"Status: {task['status']}"
                    )
            else:
                st.caption("Nenhuma tarefa registrada.")

        with st.expander("Decisões", expanded=True):
            if board["decisions"]:
                for decision in board["decisions"]:
                    reason_text = (
                        f"  \nMotivo: {decision['reason']}"
                        if decision.get("reason")
                        else ""
                    )
                    st.markdown(
                        f"**#{decision['id']}** {decision['decision']}{reason_text}"
                    )
            else:
                st.caption("Nenhuma decisão registrada.")

        with st.expander("Seções do documento", expanded=True):
            if board["document_sections"]:
                for section, data in board["document_sections"].items():
                    st.markdown(f"**{section}**  \n{data['text']}")
            else:
                st.caption("Nenhuma seção criada.")


def _render_chat_history() -> None:
    """Render user and final assistant messages in the main chat container."""
    for message in st.session_state.messages:
        role = _message_role(message)
        if role is None:
            continue

        with st.chat_message(role):
            st.markdown(message.content)


def _ensure_session_state() -> None:
    """Ensure required keys exist in Streamlit session_state on startup."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "project_board" not in st.session_state:
        st.session_state.project_board = react_agent.project_board
    if "tools_called" not in st.session_state:
        st.session_state.tools_called = set()
    if "current_user" not in st.session_state:
        st.session_state.current_user = usuarios[0]


def _bind_project_board() -> None:
    """Bind the session_state project board reference to the agent module's board."""
    st.session_state.project_board = react_agent.project_board


st.set_page_config(
    page_title="Assistente de Planejamento de Evento",
    page_icon="📅",
    layout="centered",
)

usuarios = react_agent.get_users()
_ensure_session_state()

# ============================================================================
# SEÇÃO DE INFORMAÇÕES DO TRABALHO
# ============================================================================

st.title("📅 AcademicFlow-Agent")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    **SSC0528 - Sistemas Colaborativos: Fundamentos e Aplicações**
    """)
    st.image(str(LOGO_PATH), width=200)
with col2:
    st.markdown("""
    **Alunos:**
    - Pedro Eduardo Moreira Chamone
    - Pedro Santos Souza
    """)

st.markdown("---")
st.markdown("""
Sistema colaborativo baseado em agentes de IA para auxiliar no planejamento de eventos acadêmicos.
Utilize o chat abaixo para interagir com o assistente.
""")
st.markdown("---")

_render_user_chooser()
_render_board_summary()
_render_tools_called()
_render_chat_history()

user_prompt = st.chat_input("Escreva sua mensagem para o agente...")

if user_prompt:

    # Identificar qual usuário enviou a mensagem (preferir seleção em sidebar)
    sender = st.session_state.get("current_user") or "Usuário Desconhecido"
    pending_messages = [
        *st.session_state.messages,
        HumanMessage(content=f"[user:{sender}] {user_prompt}"),
    ]

    with st.spinner("Processando com o agente..."):
        final_step = None
        try:
            for step in react_agent.react_event_agent.stream(
                {"messages": pending_messages},
                stream_mode="values"
            ):
                final_step = step
                react_agent.print_step(step)

                if final_step:
                    msg = step["messages"][-1]
                    role = getattr(msg, "type", "unknown")

                    if role == "ai" and getattr(msg, "tool_calls", None):
                        for tool_call in msg.tool_calls:
                            st.session_state.tools_called.add(tool_call["name"])

        except ConnectError:
            st.error(
                "Não consegui conectar ao Ollama. Verifique se o serviço está ativo com `ollama serve` e se o modelo `llama3.2` foi baixado."
            )
            st.stop()
        if final_step:
            st.session_state.messages = final_step["messages"]
        _bind_project_board()
        st.rerun()
