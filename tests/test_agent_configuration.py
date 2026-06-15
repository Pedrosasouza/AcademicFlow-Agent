import React_Projetct as agent


def test_available_users_are_defined_for_simulated_multiuser_flow():
    assert agent.get_users() == ["Alice", "Bob", "Charlie"]


def test_agent_tools_include_collaboration_and_coordination_actions():
    tool_names = {tool.name for tool in agent.tools}

    assert tool_names == {
        "register_task",
        "register_decision",
        "update_document_section",
        "consult_project_board",
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
