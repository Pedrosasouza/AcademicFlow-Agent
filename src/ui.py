from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

from httpx import ConnectError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.exc import OperationalError
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import agent as react_agent
from src import repository
from src.database import init_db

LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"
LIVE_REFRESH_INTERVAL = "5s"


def _get_current_session() -> str:
    """Retorna o identificador da sala/evento selecionado na sessao do Streamlit."""
    if "current_session" not in st.session_state or not st.session_state.current_session:
        st.session_state.current_session = "default"
    return str(st.session_state.current_session)


def _message_role(message: BaseMessage) -> str | None:
    """Retorna o papel no chat ('user' ou 'assistant') apenas para mensagens exibiveis."""
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        if getattr(message, "tool_calls", None) or not message.content:
            return None
        return "assistant"
    return None


def _extract_tools_called(messages: Sequence[BaseMessage]) -> set[str]:
    """Extrai o conjunto de nomes de ferramentas invocadas a partir do historico de mensagens."""
    called: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                name = tool_call.get("name")
                if name:
                    called.add(name)
    return called


def _render_tools_called() -> None:
    """Renderiza badges na barra lateral para cada ferramenta acionada na sessao."""
    with st.sidebar:
        for tool_name in sorted(st.session_state.tools_called):
            st.badge(f"Ferramenta chamada: {tool_name}", color="orange")


def _reset_session(session_id: str | None = None) -> None:
    """Limpa tarefas, decisoes, secoes e historico de chat da sessao/evento atual no banco."""
    target_session = session_id or _get_current_session()
    repository.clear_project_board(session_id=target_session)
    if "messages" in st.session_state:
        st.session_state.messages = []
    if "tools_called" in st.session_state:
        st.session_state.tools_called = set()
    if "project_board" in st.session_state:
        st.session_state.project_board = repository.get_project_board(
            session_id=target_session
        )


def _render_event_chooser() -> str:
    """Renderiza na barra lateral o seletor de evento/sala e a criacao de novas salas."""
    if "custom_sessions" not in st.session_state:
        st.session_state.custom_sessions = set()

    db_sessions = repository.list_sessions()
    all_sessions = ["default"] + sorted(
        (set(db_sessions) | set(st.session_state.custom_sessions)) - {"default"}
    )

    current = _get_current_session()
    if current not in all_sessions:
        all_sessions.append(current)

    with st.sidebar:
        st.markdown("### Evento / Sala")
        selected = st.selectbox(
            "Selecione o evento",
            all_sessions,
            index=all_sessions.index(current),
        )
        if selected != current:
            st.session_state.current_session = selected
            react_agent.set_active_session(selected)
            st.rerun()

        with st.expander("➕ Novo evento / sala"):
            new_event = st.text_input(
                "Identificador do evento",
                placeholder="Ex: semana_computacao",
            )
            if st.button("Criar / Acessar sala"):
                cleaned = new_event.strip()
                if cleaned:
                    st.session_state.custom_sessions.add(cleaned)
                    st.session_state.current_session = cleaned
                    react_agent.set_active_session(cleaned)
                    st.rerun()

    react_agent.set_active_session(selected)
    return selected


def _render_user_chooser(usuarios: list[str], session_id: str) -> None:
    """Renderiza o seletor de participante e o botao de limpeza na barra lateral."""
    with st.sidebar:
        user = st.selectbox("Selecione o usuário", usuarios)
        st.session_state.current_user = user
        if st.button("Limpar sessão"):
            _reset_session(session_id=session_id)
            st.rerun()


def _render_regulations_panel() -> None:
    """Renderiza na barra lateral a base vetorial de editais e permite indexar novas normas."""
    with st.sidebar:
        with st.expander("📚 Editais e Normas (RAG pgvector)", expanded=False):
            chunks = repository.list_regulation_chunks()
            st.caption(f"{len(chunks)} normas indexadas na base vetorial.")
            for item in chunks:
                st.markdown(f"**{item['title']}** *({item['source']})*  \n{item['content']}")
            st.markdown("---")
            source = st.text_input("Fonte / Edital", placeholder="Ex: Edital PRCEU 03/2026")
            title = st.text_input("Título da Regra", placeholder="Ex: Diárias de Palestrantes")
            content = st.text_area("Texto da Norma")
            if st.button("Indexar norma no banco"):
                if title.strip() and content.strip():
                    repository.index_regulation_chunk(
                        source=source.strip() or "Edital Institucional",
                        title=title.strip(),
                        content=content.strip(),
                    )
                    st.rerun()


def _render_board_summary() -> None:
    """Renderiza as metricas, tarefas, decisoes e secoes do quadro do projeto."""
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
    """Renderiza as mensagens do usuario (com autoria) e respostas finais do assistente no chat."""
    for message in st.session_state.messages:
        role = _message_role(message)
        if role is None:
            continue

        with st.chat_message(role):
            author = getattr(message, "name", None) if role == "user" else None
            if author:
                st.caption(f"👤 {author}")
            st.markdown(message.content)


@st.cache_resource
def _initialize_backend() -> bool:
    """Inicializa o schema e seed do banco uma unica vez no ciclo de vida do servidor Streamlit."""
    init_db()
    return True


def _sync_session_state_from_db(
    usuarios: list[str],
    session_id: str = "default",
) -> None:
    """Sincroniza o session_state do Streamlit com os dados da sala/evento no banco relacional."""
    react_agent.set_active_session(session_id)
    messages = repository.load_messages(session_id=session_id)
    st.session_state.messages = messages
    st.session_state.project_board = repository.get_project_board(session_id=session_id)
    st.session_state.tools_called = _extract_tools_called(messages)
    if "current_user" not in st.session_state and usuarios:
        st.session_state.current_user = usuarios[0]


@st.fragment(run_every=LIVE_REFRESH_INTERVAL)
def _render_live_collaboration_view(
    usuarios: list[str],
    active_session: str,
) -> None:
    """Fragmento com autorefresh periodico para sincronizar o quadro e o chat multiusuario."""
    _sync_session_state_from_db(usuarios, session_id=active_session)
    _render_board_summary()
    _render_tools_called()
    _render_chat_history()


def render_app() -> None:
    """Ponto de entrada principal da interface Streamlit."""
    st.set_page_config(
        page_title="Assistente de Planejamento de Evento",
        page_icon="📅",
        layout="centered",
    )

    st.title("📅 AcademicFlow-Agent")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
        **SSC0528 - Sistemas Colaborativos: Fundamentos e Aplicações**
        """
        )
        st.image(str(LOGO_PATH), width=200)
    with col2:
        st.markdown(
            """
        **Aluno:**
        - Pedro Santos Souza
        """
        )

    st.markdown("---")
    st.markdown(
        """
    Sistema colaborativo baseado em agentes de IA para auxiliar no planejamento de eventos acadêmicos.
    Utilize o chat abaixo para interagir com o assistente.
    """
    )
    st.markdown("---")

    try:
        _initialize_backend()
        usuarios = react_agent.get_users()
        active_session = _render_event_chooser()
        _sync_session_state_from_db(usuarios, session_id=active_session)
    except OperationalError:
        st.error(
            "Não foi possível conectar ao banco de dados PostgreSQL. "
            "Verifique se o container está em execução com `docker compose up -d` "
            "e confira a variável `DATABASE_URL` no arquivo `.env`."
        )
        st.stop()

    _render_user_chooser(usuarios, session_id=active_session)
    _render_regulations_panel()
    _render_live_collaboration_view(usuarios, active_session=active_session)

    user_prompt = st.chat_input("Escreva sua mensagem para o agente...")

    if user_prompt:
        sender = st.session_state.get("current_user") or "Usuário Desconhecido"

        with st.spinner("Processando com o agente..."):
            try:
                react_agent.run_turn(
                    user_input=user_prompt,
                    session_id=active_session,
                    author=sender,
                )
            except ConnectError:
                st.error(
                    "Não consegui conectar ao Ollama. Verifique se o serviço está ativo com `ollama serve` e se o modelo `llama3.2` foi baixado."
                )
                st.stop()
            except OperationalError:
                st.error(
                    "Falha de comunicação com o banco de dados PostgreSQL durante o processamento."
                )
                st.stop()

            _sync_session_state_from_db(usuarios, session_id=active_session)
            st.rerun()


if __name__ == "__main__" or get_script_run_ctx() is not None:
    render_app()
