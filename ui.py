import streamlit as st
import React_Projetct as react_agent
from langchain_core.messages import AIMessage, HumanMessage
from httpx import ConnectError

def _message_role(message) -> str | None:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        if getattr(message, "tool_calls", None) and not message.content:
            return None
        return "assistant"
    return None

def _render_tools_called():
    with st.sidebar:
        for tool_name in st.session_state.tools_called:
            st.badge(f"Ferramenta chamada: {tool_name}", color="orange")

def _render_user_chooser():
    with st.sidebar:
        user = st.selectbox("Selecione o usuário", usuarios)
        st.session_state.current_user = user

def _render_board_summary() -> None:
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
                    )

def _render_chat_history() -> None:
    for message in st.session_state.messages:
        role = _message_role(message)
        if role is None:
            continue

        with st.chat_message(role):
            st.markdown(message.content)

def _fresh_board() -> dict:
    return {
        "tasks": [],
        "decisions": [],
        "document_sections": {},
    }

def _ensure_session_state() -> None:
    '''Garantir que as chaves necessárias existam no session_state quando a aplicação iniciar.'''
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "project_board" not in st.session_state:
        st.session_state.project_board = _fresh_board()
    if "tools_called" not in st.session_state:
        st.session_state.tools_called = set()
    if "current_user" not in st.session_state:
        st.session_state.current_user = usuarios[0]

def _bind_project_board() -> None:
    '''Vincula o dicionário do quadro do projeto na sessão do Streamlit à variável usada pelas ferramentas do agente.'''
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
    st.image("logo.png", width=200)
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

user_prompt = st.chat_input("Escreva sua mensagem para o agente...")

if user_prompt:

    # Identificar qual usuário enviou a mensagem (preferir seleção em sidebar)
    sender = st.session_state.get("current_user") or "Usuário Desconecido"
    st.session_state.messages.append(HumanMessage(content=f"[user:{sender}] {user_prompt}"))

    with st.spinner("Processando com o agente..."):
        final_step = None
        try:
            for step in react_agent.react_event_agent.stream(
                {"messages": st.session_state.messages},
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
            st.session_state.messages.append(AIMessage(content=final_step["messages"][-1].content))
        _bind_project_board()
        st.rerun()

_render_board_summary()
_render_chat_history()
_render_tools_called()
_render_user_chooser()
