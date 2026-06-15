import React_Project as agent


def test_register_task_adds_task_with_given_fields():
    result = agent.register_task.invoke(
        {
            "title": "Reservar auditorio",
            "owner": "Alice",
            "deadline": "sexta-feira",
        }
    )

    assert "Tarefa #1 registrada" in result
    assert len(agent.project_board["tasks"]) == 1
    assert agent.project_board["tasks"][0]["title"] == "Reservar auditorio"
    assert agent.project_board["tasks"][0]["owner"] == "Alice"
    assert agent.project_board["tasks"][0]["deadline"] == "sexta-feira"
    assert agent.project_board["tasks"][0]["status"] == "pendente"


def test_register_task_uses_default_owner_and_deadline():
    agent.register_task.invoke({"title": "Definir equipe de recepcao"})

    task = agent.project_board["tasks"][0]
    assert task["owner"] == "a definir"
    assert task["deadline"] == "a definir"


def test_register_decision_adds_decision_with_reason():
    result = agent.register_decision.invoke(
        {
            "decision": "Realizar o evento no periodo da tarde",
            "reason": "Maior disponibilidade dos participantes",
        }
    )

    assert "Decisao #1 registrada" in result
    assert len(agent.project_board["decisions"]) == 1
    assert agent.project_board["decisions"][0]["decision"] == (
        "Realizar o evento no periodo da tarde"
    )
    assert agent.project_board["decisions"][0]["reason"] == (
        "Maior disponibilidade dos participantes"
    )


def test_update_document_section_creates_or_replaces_section():
    first_result = agent.update_document_section.invoke(
        {
            "section": "Objetivos",
            "text": "Organizar uma semana academica interdisciplinar.",
        }
    )
    second_result = agent.update_document_section.invoke(
        {
            "section": "Objetivos",
            "text": "Organizar um evento academico com palestras e oficinas.",
        }
    )

    assert "Secao 'Objetivos' atualizada" in first_result
    assert "Secao 'Objetivos' atualizada" in second_result
    assert len(agent.project_board["document_sections"]) == 1
    assert agent.project_board["document_sections"]["Objetivos"]["text"] == (
        "Organizar um evento academico com palestras e oficinas."
    )


def test_consult_project_board_returns_current_state_summary():
    agent.register_task.invoke(
        {
            "title": "Enviar convites",
            "owner": "Bob",
            "deadline": "segunda-feira",
        }
    )
    agent.register_decision.invoke(
        {
            "decision": "Usar o auditorio principal",
            "reason": "Maior capacidade",
        }
    )
    agent.update_document_section.invoke(
        {
            "section": "Programacao",
            "text": "Abertura, palestra principal e mesa redonda.",
        }
    )

    summary = agent.consult_project_board.invoke({})

    assert "Tarefas:" in summary
    assert "#1 Enviar convites | Bob | segunda-feira | pendente" in summary
    assert "Decisoes:" in summary
    assert "#1 Usar o auditorio principal" in summary
    assert "Secoes do documento:" in summary
    assert "Programacao" in summary


def test_consult_project_board_handles_empty_state():
    summary = agent.consult_project_board.invoke({})

    assert "- nenhuma tarefa registrada" in summary
    assert "- nenhuma decisao registrada" in summary
    assert "- nenhuma secao criada" in summary
