from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src import agent


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
                        content="[user:Alice] Crie uma tarefa para reservar o auditorio ate sexta-feira."
                    )
                ]
            }
        )

    assert mock_llm.invoke.call_count == 2
    assert len(agent.project_board["tasks"]) == 1
    assert agent.project_board["tasks"][0]["title"] == "Reservar auditorio"
    assert agent.project_board["tasks"][0]["owner"] == "Alice"
    assert agent.project_board["tasks"][0]["deadline"] == "sexta-feira"

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
            {"messages": [HumanMessage(content="[user:Bob] Ola!")]}
        )

    assert mock_llm.invoke.call_count == 1
    assert len(result["messages"]) == 2
    assert result["messages"][-1].content == direct_reply.content
