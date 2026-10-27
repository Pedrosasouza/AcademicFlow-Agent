"""
Testes automatizados para recuperacao de falhas no turno (resiliencia do agente ReAct).
Cobre o cenario em que a ferramenta altera dados no banco, mas o LLM falha antes da resposta final,
garantindo persistencia do historico parcial com aviso de recuperacao e idempotencia contra duplicacao.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from httpx import ConnectError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

from src import agent, repository


def test_run_turn_persists_partial_history_and_recovery_message_when_llm_fails_after_tool() -> None:
    """Garante que se o LLM falhar apos a tool gravar no banco, o historico parcial e o aviso sao persistidos."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "register_task",
                "args": {
                    "title": "Reservar auditorio principal",
                    "owner": "Alice",
                    "deadline": "2026-11-20",
                },
                "id": "call_resilience_1",
                "type": "tool_call",
            }
        ],
    )

    mock_llm = MagicMock()
    # 1a chamada: LLM decide chamar a tool
    # 2a chamada (apos a execucao da tool): LLM cai com erro de conexao
    mock_llm.invoke.side_effect = [
        tool_call_msg,
        ConnectError("Conexao com o Ollama perdida durante a sintese final"),
    ]

    compiled_agent = agent.build_agent(custom_llm=mock_llm)

    with pytest.raises(ConnectError):
        agent.run_turn(
            user_input="Por favor, reserve o auditorio principal para dia 20.",
            session_id="sala_resiliencia",
            author="Alice",
            compiled_agent=compiled_agent,
        )

    # 1. A tarefa foi de fato gravada no banco de dados pela tool
    board = repository.get_project_board(session_id="sala_resiliencia")
    assert len(board["tasks"]) == 1
    assert board["tasks"][0]["title"] == "Reservar auditorio principal"
    assert board["tasks"][0]["owner"] == "Alice"

    # 2. O historico do turno nao foi descartado: deve conter HumanMessage, AIMessage, ToolMessage e recovery AIMessage
    messages = repository.load_messages(session_id="sala_resiliencia")
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Por favor, reserve o auditorio principal para dia 20."
    assert messages[0].name == "Alice"

    assert isinstance(messages[1], AIMessage)
    assert messages[1].tool_calls[0]["name"] == "register_task"

    assert isinstance(messages[2], ToolMessage)
    assert "Tarefa #1 registrada" in messages[2].content

    assert isinstance(messages[3], AIMessage)
    assert "A operacao foi executada e registrada no banco de dados" in messages[3].content


def test_subsequent_turn_loads_recovered_context_safely_without_orphan_tool_message() -> None:
    """Garante que apos o turno com falha recuperada, o turno seguinte carrega contexto integro."""
    # Simula estado anterior persistido com recovery
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "register_decision",
                "args": {"decision": "Entrada gratuita"},
                "id": "call_dec_rec_1",
            }
        ],
    )
    mock_llm_fail = MagicMock()
    mock_llm_fail.invoke.side_effect = [tool_call_msg, ConnectError("Ollama caiu")]
    failing_graph = agent.build_agent(custom_llm=mock_llm_fail)

    with pytest.raises(ConnectError):
        agent.run_turn(
            user_input="Decidimos que a entrada sera gratuita.",
            session_id="sala_rec_next",
            author="Bob",
            compiled_agent=failing_graph,
        )

    # Agora a conexao volta ao normal e o usuario manda outra mensagem
    mock_llm_recovered = MagicMock()
    mock_llm_recovered.invoke.return_value = AIMessage(
        content="No momento, temos 1 decisao registrada de entrada gratuita."
    )
    recovered_graph = agent.build_agent(custom_llm=mock_llm_recovered)

    new_msgs = agent.run_turn(
        user_input="Qual o resumo do projeto?",
        session_id="sala_rec_next",
        author="Charlie",
        compiled_agent=recovered_graph,
    )

    assert len(new_msgs) == 2
    assert isinstance(new_msgs[0], HumanMessage)
    assert isinstance(new_msgs[1], AIMessage)

    # Contexto total deve conter o turno anterior com recuperacao e o novo turno
    all_msgs = repository.load_messages(session_id="sala_rec_next")
    assert len(all_msgs) == 6  # 4 do primeiro turno + 2 do segundo
    # A ultima mensagem da janela de contexto enviada ao LLM deve ser a nova
    window = repository.load_context_messages(session_id="sala_rec_next")
    assert not isinstance(window[0], ToolMessage)


def test_retry_after_failure_does_not_create_duplicate_task() -> None:
    """Garante que a repeticao da solicitacao apos falha nao duplica a tarefa no banco (idempotencia)."""
    # 1. Primeira execucao com falha apos criar a tarefa
    mock_llm_fail = MagicMock()
    mock_llm_fail.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "register_task",
                    "args": {
                        "title": "Comprar crachas",
                        "owner": "Alice",
                        "deadline": "2026-11-15",
                    },
                    "id": "c1",
                }
            ],
        ),
        ConnectError("timeout"),
    ]
    graph_fail = agent.build_agent(custom_llm=mock_llm_fail)

    with pytest.raises(ConnectError):
        agent.run_turn(
            user_input="Crie uma tarefa para comprar crachas para Alice.",
            session_id="sala_dedup",
            author="Alice",
            compiled_agent=graph_fail,
        )

    tasks_initial = repository.get_project_board(session_id="sala_dedup")["tasks"]
    assert len(tasks_initial) == 1
    assert tasks_initial[0]["title"] == "Comprar crachas"

    # 2. Usuario repete a mesma ordem exatamente igual
    mock_llm_retry = MagicMock()
    mock_llm_retry.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "register_task",
                    "args": {
                        "title": "Comprar crachas",
                        "owner": "Alice",
                        "deadline": "2026-11-15",
                    },
                    "id": "c2",
                }
            ],
        ),
        AIMessage(content="A tarefa de crachas ja estava registrada para Alice."),
    ]
    graph_retry = agent.build_agent(custom_llm=mock_llm_retry)

    agent.run_turn(
        user_input="Crie uma tarefa para comprar crachas para Alice.",
        session_id="sala_dedup",
        author="Alice",
        compiled_agent=graph_retry,
    )

    # O numero total de tarefas no banco continua sendo exatamente 1
    tasks_after_retry = repository.get_project_board(session_id="sala_dedup")["tasks"]
    assert len(tasks_after_retry) == 1
    assert tasks_after_retry[0]["id"] == tasks_initial[0]["id"]


def test_decision_and_document_section_safe_repetition() -> None:
    """Valida a deduplicacao de decisoes repetidas e o upsert de secoes do documento."""
    d1 = repository.create_decision("Usar projetor 4K", reason="Qualidade", session_id="sala_safe")
    d2 = repository.create_decision("Usar projetor 4K", reason="Nova justificativa", session_id="sala_safe")

    assert d1["id"] == d2["id"]
    assert d2.get("is_duplicate") is True
    decisions = repository.list_decisions(session_id="sala_safe")
    assert len(decisions) == 1

    # Permissao explicita para duplicar se solicitado
    d3 = repository.create_decision("Usar projetor 4K", session_id="sala_safe", allow_duplicate=True)
    assert d3["id"] != d1["id"]
    assert len(repository.list_decisions(session_id="sala_safe")) == 2

    # Secoes do documento continuam sendo upsert por definicao
    s1 = repository.upsert_document_section("Introducao", "Texto v1", session_id="sala_safe")
    s2 = repository.upsert_document_section("Introducao", "Texto v2", session_id="sala_safe")
    sections = repository.get_document_sections(session_id="sala_safe")
    assert len(sections) == 1
    assert sections["Introducao"]["text"] == "Texto v2"
