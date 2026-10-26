from unittest.mock import MagicMock, patch

from httpx import ConnectError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy.exc import OperationalError

from src import agent, repository


def test_available_users_are_defined_for_simulated_multiuser_flow():
    assert agent.get_users() == ["Alice", "Bob", "Charlie"]


def test_agent_tools_include_collaboration_and_coordination_actions():
    tool_names = {tool.name for tool in agent.tools}

    assert tool_names == {
        "register_task",
        "update_task",
        "register_decision",
        "update_document_section",
        "consult_project_board",
        "search_event_regulations",
    }


def test_system_prompt_mentions_collaborative_agent_role():
    prompt = agent.SYSTEM_PROMPT.lower()

    assert "agente colaborativo" in prompt
    assert "planejamento de um evento academico" in prompt
    assert "tarefas" in prompt
    assert "decisoes" in prompt


def test_langgraph_app_is_compiled_and_callable():
    assert agent.react_event_agent is not None
    assert callable(agent.react_event_agent.invoke)
    assert callable(agent.react_event_agent.stream)


def test_react_graph_executes_tool_call_cycle_without_ollama():
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "register_task",
                "args": {
                    "title": "Reservar auditorio",
                    "owner": "Alice",
                    "deadline": "sexta-feira",
                },
                "id": "call_register_1",
                "type": "tool_call",
            }
        ],
    )
    final_ai_msg = AIMessage(
        content="Tarefa #1 registrada para Alice com prazo sexta-feira."
    )

    with patch.object(agent, "llm") as mock_llm:
        mock_llm.invoke.side_effect = [tool_call_msg, final_ai_msg]

        result = agent.react_event_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Crie uma tarefa para reservar o auditorio ate sexta-feira.",
                        name="Alice",
                    )
                ]
            }
        )

    assert mock_llm.invoke.call_count == 2
    tasks = repository.get_project_board()["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Reservar auditorio"
    assert tasks[0]["owner"] == "Alice"
    assert tasks[0]["deadline"] == "sexta-feira"

    messages = result["messages"]
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage) and messages[1].tool_calls
    assert isinstance(messages[2], ToolMessage)
    assert "Tarefa #1 registrada" in messages[2].content
    assert isinstance(messages[3], AIMessage)
    assert messages[3].content == final_ai_msg.content


def test_react_graph_answers_directly_when_no_tool_is_needed():
    direct_reply = AIMessage(content="Ola! Como posso ajudar no planejamento hoje?")

    with patch.object(agent, "llm") as mock_llm:
        mock_llm.invoke.return_value = direct_reply

        result = agent.react_event_agent.invoke(
            {"messages": [HumanMessage(content="Ola!", name="Bob")]}
        )

    assert mock_llm.invoke.call_count == 1
    assert len(result["messages"]) == 2
    assert result["messages"][-1].content == direct_reply.content


def test_run_turn_and_build_agent_persist_clean_author_without_user_prefix():
    """Garante que run_turn e build_agent persistem o turno com author/user_id limpos."""
    custom_llm = MagicMock()
    custom_llm.invoke.return_value = AIMessage(content="Entendido, Alice!")
    isolated_graph = agent.build_agent(custom_llm=custom_llm)

    new_msgs = agent.run_turn(
        user_input="Vamos definir o cronograma.",
        session_id="sala_teste",
        author="Alice",
        compiled_agent=isolated_graph,
    )

    assert len(new_msgs) == 2
    saved_history = repository.load_messages(session_id="sala_teste")
    assert len(saved_history) == 2
    assert isinstance(saved_history[0], HumanMessage)
    assert saved_history[0].content == "Vamos definir o cronograma."
    assert saved_history[0].name == "Alice"


def test_run_turn_invokes_print_step_callback_for_tool_call_and_final_answer(capsys):
    """Garante que print_step exibe no terminal a chamada da ferramenta, o resultado e a resposta final."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "register_decision",
                "args": {"decision": "Evento gratuito", "reason": "Inclusao"},
                "id": "call_dec_1",
                "type": "tool_call",
            }
        ],
    )
    final_msg = AIMessage(content="Decisao registrada com sucesso.")

    custom_llm = MagicMock()
    custom_llm.invoke.side_effect = [tool_call_msg, final_msg]
    isolated_graph = agent.build_agent(custom_llm=custom_llm)

    agent.run_turn(
        user_input="Registre a decisao de evento gratuito.",
        session_id="default",
        author="Bob",
        on_step=agent.print_step,
        compiled_agent=isolated_graph,
    )

    output = capsys.readouterr().out
    assert "[chamando ferramenta: register_decision]" in output
    assert "[resultado da ferramenta: Decisao #1 registrada: Evento gratuito. Motivo: Inclusao]" in output
    assert "Agente: Decisao registrada com sucesso." in output


def test_cli_main_executes_interactive_turn_and_exits_cleanly(monkeypatch, capsys):
    """Valida o loop interativo main() da CLI processando um turno real e encerrando com 'exit'."""
    inputs = iter(["Ola equipe", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with patch.object(agent, "llm") as mock_llm:
        mock_llm.invoke.return_value = AIMessage(content="Ola! Pronto para ajudar.")
        agent.main()

    output = capsys.readouterr().out
    assert "=== React Agent Colaborativo: Planejamento de Evento ===" in output
    assert "Agente: Ola! Pronto para ajudar." in output
    assert len(repository.load_messages(session_id="default")) == 2


def test_cli_main_handles_database_and_ollama_errors_gracefully(monkeypatch, capsys):
    """Garante que main() exibe mensagens claras quando o PostgreSQL ou o Ollama falham."""
    # 1. Falha de banco logo no init_db()
    with patch.object(
        agent,
        "init_db",
        side_effect=OperationalError("SELECT 1", {}, Exception("DB down")),
    ):
        agent.main()

    out_db_init = capsys.readouterr().out
    assert "Nao consegui conectar ao banco de dados PostgreSQL." in out_db_init

    # 2. Falha de conexao com Ollama durante run_turn()
    monkeypatch.setattr("builtins.input", lambda _: "Pergunta ao agente")
    with patch.object(
        agent,
        "run_turn",
        side_effect=ConnectError("Ollama offline"),
    ):
        agent.main()

    out_ollama = capsys.readouterr().out
    assert "Nao consegui conectar ao Ollama." in out_ollama

    # 3. Queda do banco durante run_turn()
    with patch.object(
        agent,
        "run_turn",
        side_effect=OperationalError("INSERT", {}, Exception("DB lost")),
    ):
        agent.main()

    out_db_turn = capsys.readouterr().out
    assert "Erro de conexao com o banco de dados durante a execucao." in out_db_turn


def test_search_event_regulations_returns_message_when_no_matches_found(monkeypatch):
    """Garante que search_event_regulations informa claramente quando nenhuma norma e encontrada."""
    monkeypatch.setattr(repository, "search_regulation_chunks", lambda query, top_k=3: [])
    result = agent.search_event_regulations.invoke({"query": "assunto inexistente"})
    assert result == "Nenhuma norma ou edital encontrado para a consulta."
