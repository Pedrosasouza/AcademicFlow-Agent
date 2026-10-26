"""
Testes automatizados para a interface Streamlit (src/ui.py), validando renderizacao,
interacoes de usuario, painel de editais RAG e tratamento de erros de conexao.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from httpx import ConnectError
from langchain_core.messages import AIMessage, HumanMessage
import pytest
from sqlalchemy.exc import OperationalError

from src import agent, repository, ui


class _AttrDict(dict[str, Any]):
    """Simula o st.session_state com acesso tanto por chave quanto por atributo."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class _ReusableCtx:
    """Context manager reutilizavel para simular st.sidebar, st.expander, st.chat_message e st.spinner."""

    def __enter__(self) -> _ReusableCtx:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


def _build_fake_st(
    chat_input_value: str | None = None,
    button_clicks: set[str] | None = None,
    text_inputs: dict[str, str] | None = None,
    text_areas: dict[str, str] | None = None,
    selectbox_choice: dict[str, str] | None = None,
) -> MagicMock:
    """Cria um mock configuravel do modulo streamlit para testar funcoes de UI sem servidor HTTP."""
    fake_st = MagicMock()
    fake_st.session_state = _AttrDict()
    fake_st.sidebar = _ReusableCtx()
    fake_st.expander.side_effect = lambda *a, **kw: _ReusableCtx()
    fake_st.chat_message.side_effect = lambda *a, **kw: _ReusableCtx()
    fake_st.spinner.side_effect = lambda *a, **kw: _ReusableCtx()

    def make_cols(spec: Any):
        count = spec if isinstance(spec, int) else len(spec)
        cols = []
        for _ in range(count):
            c = MagicMock()
            c.__enter__.return_value = c
            c.__exit__.return_value = False
            cols.append(c)
        return cols

    fake_st.columns.side_effect = make_cols
    fake_st.chat_input.return_value = chat_input_value

    clicks = button_clicks or set()
    fake_st.button.side_effect = lambda label, *a, **kw: label in clicks

    inputs = text_inputs or {}
    fake_st.text_input.side_effect = lambda label, *a, **kw: inputs.get(label, "")

    areas = text_areas or {}
    fake_st.text_area.side_effect = lambda label, *a, **kw: areas.get(label, "")

    choices = selectbox_choice or {}

    def selectbox(label: str, options: list[str], index: int = 0, **kw: Any) -> str:
        if label in choices and choices[label] in options:
            return choices[label]
        return options[index] if options else ""

    fake_st.selectbox.side_effect = selectbox
    return fake_st


def test_ui_render_board_summary_and_chat_history_with_populated_and_empty_states(monkeypatch):
    """Valida a renderizacao do quadro lateral (vazio e populado), badges e historico de chat."""
    fake_st = _build_fake_st()
    monkeypatch.setattr(ui, "st", fake_st)

    # 1. Estado vazio
    ui._sync_session_state_from_db(["Alice", "Bob"], session_id="default")
    ui._render_board_summary()
    ui._render_chat_history()
    ui._render_tools_called()

    # 2. Estado populado com tarefas, decisoes, secoes e mensagens com autoria
    repository.create_task("Contratar coffee break", owner="Alice", deadline="2026-11-20")
    repository.create_decision("Evento hibrido", reason="Alcance nacional")
    repository.upsert_document_section("Resumo", "Texto da secao de resumo.")
    repository.append_messages(
        [
            HumanMessage(content="Vamos contratar o coffee break", name="Alice"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "register_task", "args": {}, "id": "c1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="Tarefa criada com sucesso!"),
        ]
    )

    ui._sync_session_state_from_db(["Alice", "Bob"], session_id="default")
    ui._render_board_summary()
    ui._render_chat_history()
    ui._render_tools_called()

    assert fake_st.badge.called
    assert any("👤 Alice" in str(call.args[0]) for call in fake_st.caption.call_args_list)


def test_ui_event_chooser_user_chooser_and_regulations_panel(monkeypatch):
    """Valida a criacao de nova sala, limpeza de sessao e indexacao de norma RAG pela barra lateral."""
    fake_st = _build_fake_st(
        button_clicks={"Criar / Acessar sala", "Limpar sessão", "Indexar norma no banco"},
        text_inputs={
            "Identificador do evento": "simposio_2026",
            "Fonte / Edital": "Edital USP 99/2026",
            "Título da Regra": "Norma de Patrocinio",
        },
        text_areas={"Texto da Norma": "Regras para patrocinio de eventos academicos."},
        selectbox_choice={"Selecione o usuário": "Bob"},
    )
    monkeypatch.setattr(ui, "st", fake_st)

    ui._render_event_chooser()
    assert fake_st.session_state.current_session == "simposio_2026"
    assert "simposio_2026" in fake_st.session_state.custom_sessions

    ui._render_user_chooser(["Alice", "Bob"], session_id="simposio_2026")
    assert fake_st.session_state.current_user == "Bob"

    initial_chunks = len(repository.list_regulation_chunks())
    ui._render_regulations_panel()
    assert len(repository.list_regulation_chunks()) == initial_chunks + 1


def test_ui_render_app_executes_chat_turn_and_handles_connection_errors(monkeypatch):
    """Valida o fluxo completo de render_app() no envio de mensagem e tratamento de ConnectError/OperationalError."""
    fake_st = _build_fake_st(chat_input_value="Quais as tarefas atuais?")
    monkeypatch.setattr(ui, "st", fake_st)
    monkeypatch.setattr(ui, "_initialize_backend", lambda: True)
    monkeypatch.setattr(
        ui,
        "_render_live_collaboration_view",
        lambda usuarios, active_session: (
            ui._sync_session_state_from_db(usuarios, session_id=active_session),
            ui._render_board_summary(),
            ui._render_tools_called(),
            ui._render_chat_history(),
        ),
    )

    called_turns: list[dict[str, Any]] = []

    def fake_run_turn(**kwargs: Any):
        called_turns.append(kwargs)
        return [HumanMessage(content=kwargs["user_input"], name=kwargs["author"])]

    monkeypatch.setattr(agent, "run_turn", fake_run_turn)

    ui.render_app()
    assert len(called_turns) == 1
    assert called_turns[0]["user_input"] == "Quais as tarefas atuais?"
    assert called_turns[0]["author"] == "Alice"

    # Simula falha de conexao com o Ollama durante o turno
    class _StopCalled(Exception):
        pass

    fake_st.stop.side_effect = _StopCalled
    monkeypatch.setattr(
        agent,
        "run_turn",
        MagicMock(side_effect=ConnectError("Ollama offline")),
    )
    with pytest.raises(_StopCalled):
        ui.render_app()
    assert fake_st.error.called

    # Simula falha de conexao com o PostgreSQL durante o turno
    monkeypatch.setattr(
        agent,
        "run_turn",
        MagicMock(side_effect=OperationalError("INSERT", {}, Exception("DB lost"))),
    )
    with pytest.raises(_StopCalled):
        ui.render_app()

    # Simula falha de conexao com o PostgreSQL na inicializacao
    monkeypatch.setattr(
        ui,
        "_initialize_backend",
        MagicMock(side_effect=OperationalError("SELECT 1", {}, Exception("DB down"))),
    )
    with pytest.raises(_StopCalled):
        ui.render_app()
