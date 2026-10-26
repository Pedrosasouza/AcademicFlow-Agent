from src import agent, repository


def test_tools_persist_directly_to_database_single_source_of_truth():
    """Garante que as tools gravam no banco via repository como fonte unica da verdade."""
    agent.register_task.invoke(
        {
            "title": "Validar persistencia no banco",
            "owner": "Alice",
            "deadline": "2026-11-15",
        }
    )
    agent.register_decision.invoke(
        {
            "decision": "Usar PostgreSQL",
            "reason": "Persistencia compartilhada",
        }
    )
    agent.update_document_section.invoke(
        {
            "section": "Infraestrutura",
            "text": "Banco relacional com SQLAlchemy 2.0.",
        }
    )

    db_board = repository.get_project_board()
    assert len(db_board["tasks"]) == 1
    assert db_board["tasks"][0]["title"] == "Validar persistencia no banco"
    assert db_board["tasks"][0]["owner_id"] is not None
    assert db_board["tasks"][0]["deadline"] == "2026-11-15"
    assert len(db_board["decisions"]) == 1
    assert "Infraestrutura" in db_board["document_sections"]


def test_register_task_adds_task_with_given_fields():
    result = agent.register_task.invoke(
        {
            "title": "Reservar auditorio",
            "owner": "Alice",
            "deadline": "sexta-feira",
        }
    )

    tasks = repository.get_project_board()["tasks"]
    assert "Tarefa #1 registrada" in result
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Reservar auditorio"
    assert tasks[0]["owner"] == "Alice"
    assert tasks[0]["owner_id"] is not None
    assert tasks[0]["deadline"] == "sexta-feira"
    assert tasks[0]["status"] == "pendente"


def test_register_task_uses_default_owner_and_deadline():
    agent.register_task.invoke({"title": "Definir equipe de recepcao"})

    task = repository.get_project_board()["tasks"][0]
    assert task["owner"] == "a definir"
    assert task["owner_id"] is None
    assert task["deadline"] == "a definir"


def test_register_task_infers_active_user_from_context_when_owner_omitted_or_self():
    """Garante que register_task atribui a tarefa ao usuario ativo na ContextVar se owner nao for informado ou for 'eu'."""
    agent.set_active_user("Charlie")

    agent.register_task.invoke({"title": "Preparar slides de abertura"})
    agent.register_task.invoke({"title": "Testar projetor", "owner": "eu"})

    tasks = repository.get_project_board()["tasks"]
    assert len(tasks) == 2
    assert tasks[0]["owner"] == "Charlie"
    assert tasks[0]["owner_id"] is not None
    assert tasks[1]["owner"] == "Charlie"
    assert tasks[1]["owner_id"] is not None


def test_update_task_modifies_existing_task_fields():
    agent.register_task.invoke(
        {
            "title": "Reservar auditorio",
            "owner": "Alice",
            "deadline": "sexta-feira",
        }
    )

    result = agent.update_task.invoke(
        {
            "task_id": 1,
            "status": "concluida",
            "owner": "Bob",
            "deadline": "quinta-feira",
        }
    )

    task = repository.get_project_board()["tasks"][0]
    assert "Tarefa #1 atualizada" in result
    assert task["status"] == "concluida"
    assert task["owner"] == "Bob"
    assert task["owner_id"] is not None
    assert task["deadline"] == "quinta-feira"


def test_update_task_rejects_invalid_status_and_invalid_iso_deadline():
    """Garante que status fora do enum TaskStatus e datas ISO invalidas sao rejeitados."""
    agent.register_task.invoke({"title": "Validar status e prazo"})

    bad_status = agent.update_task.invoke({"task_id": 1, "status": "inexistente"})
    assert "Erro de validacao" in bad_status

    bad_date = agent.register_task.invoke(
        {"title": "Prazo invalido", "deadline": "2026-02-30"}
    )
    assert "Erro de validacao" in bad_date


def test_update_task_returns_not_found_for_invalid_id():
    result = agent.update_task.invoke({"task_id": 99, "status": "concluida"})

    assert result == "Tarefa #99 nao encontrada."


def test_register_decision_adds_decision_with_reason():
    result = agent.register_decision.invoke(
        {
            "decision": "Realizar o evento no periodo da tarde",
            "reason": "Maior disponibilidade dos participantes",
        }
    )

    decisions = repository.get_project_board()["decisions"]
    assert "Decisao #1 registrada" in result
    assert len(decisions) == 1
    assert decisions[0]["decision"] == (
        "Realizar o evento no periodo da tarde"
    )
    assert decisions[0]["reason"] == (
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

    sections = repository.get_project_board()["document_sections"]
    assert "Secao 'Objetivos' atualizada" in first_result
    assert "Secao 'Objetivos' atualizada" in second_result
    assert len(sections) == 1
    assert sections["Objetivos"]["text"] == (
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


def test_agent_tools_respect_active_session_context():
    """Garante que as ferramentas ReAct operam de forma isolada na sessao/evento ativo."""
    agent.set_active_session("simposio_ia")
    assert agent.get_active_session() == "simposio_ia"

    agent.register_task.invoke({"title": "Convidar palestrante de IA", "owner": "Alice"})
    agent.register_decision.invoke({"decision": "Foco em LLMs locais", "reason": "Privacidade"})
    agent.update_document_section.invoke({"section": "Tema", "text": "Agentes autonomos ReAct"})

    board_simposio = repository.get_project_board(session_id="simposio_ia")
    assert len(board_simposio["tasks"]) == 1
    assert board_simposio["tasks"][0]["title"] == "Convidar palestrante de IA"
    assert "Convidar palestrante de IA" in agent.consult_project_board.invoke({})

    # Ao voltar para a sessao 'default', o quadro deve estar completamente vazio
    agent.set_active_session("default")
    board_default = repository.get_project_board(session_id="default")
    assert board_default["tasks"] == []
    assert board_default["decisions"] == []
    assert board_default["document_sections"] == {}
    assert "- nenhuma tarefa registrada" in agent.consult_project_board.invoke({})


def test_search_event_regulations_returns_relevant_university_rules():
    """Garante que a ferramenta RAG consulta as normas/editais da universidade por similaridade vetorial."""
    result = agent.search_event_regulations.invoke(
        {"query": "qual a antecedencia para reserva de auditorio e regras de coffee break?"}
    )

    assert "Normas e editais encontrados" in result
    assert "Edital" in result or "Regimento" in result or "Auditorio" in result or "auditorio" in result.lower()
