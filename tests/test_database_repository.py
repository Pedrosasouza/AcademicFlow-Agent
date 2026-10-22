from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest
from sqlalchemy import inspect, select

from src import database, models, repository


def test_init_db_creates_all_tables_and_seeds_default_users(monkeypatch):
    """Garante que init_db cria as 5 tabelas relacionais e popula os usuarios de forma idempotente."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    database.reset_engine()

    database.init_db()
    # Chamada repetida deve ser idempotente (nao duplicar usuarios nem falhar)
    database.init_db()

    inspector = inspect(database.get_engine())
    tables = set(inspector.get_table_names())
    assert {
        "users",
        "tasks",
        "decisions",
        "document_sections",
        "chat_messages",
    }.issubset(tables)

    with database.get_db_session() as session:
        users = session.scalars(
            select(models.UserModel.name).order_by(models.UserModel.id)
        ).all()

    assert users == ["Alice", "Bob", "Charlie"]


def test_get_db_session_rolls_back_on_exception(monkeypatch):
    """Garante que uma transacao interrompida por erro faz rollback e nao corrompe o estado."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    database.reset_engine()
    database.init_db()

    with pytest.raises(RuntimeError, match="falha simulada"):
        with database.get_db_session() as session:
            session.add(
                models.TaskModel(
                    title="Tarefa que deve sofrer rollback",
                    owner="Alice",
                    deadline="amanha",
                    status="pendente",
                    created_at="2026-10-22 14:00",
                )
            )
            raise RuntimeError("falha simulada")

    with database.get_db_session() as session:
        tasks = session.scalars(select(models.TaskModel)).all()

    assert tasks == []


def test_orm_models_persist_all_fields_and_json_tool_calls(monkeypatch):
    """Valida os tipos e mapeamentos dos modelos SQLAlchemy 2.0 (incluindo JSON em ChatMessageModel)."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    database.reset_engine()
    database.init_db()

    sample_tool_calls = [
        {
            "name": "register_task",
            "args": {"title": "Reservar sala"},
            "id": "call_1",
            "type": "tool_call",
        }
    ]

    with database.get_db_session() as session:
        task = models.TaskModel(
            title="Reservar sala",
            owner="Alice",
            deadline="sexta-feira",
            status="pendente",
            created_at="2026-10-22 14:00",
        )
        decision = models.DecisionModel(
            decision="Evento presencial",
            reason="Melhor engajamento",
            created_at="2026-10-22 14:01",
        )
        section = models.DocumentSectionModel(
            section="Cronograma",
            text="Abertura as 14h",
            updated_at="2026-10-22 14:02",
        )
        message = models.ChatMessageModel(
            session_id="default",
            role="ai",
            content="",
            tool_calls=sample_tool_calls,
            tool_call_id=None,
        )
        session.add_all([task, decision, section, message])

    with database.get_db_session() as session:
        saved_task = session.scalar(select(models.TaskModel))
        saved_decision = session.scalar(select(models.DecisionModel))
        saved_section = session.scalar(select(models.DocumentSectionModel))
        saved_message = session.scalar(select(models.ChatMessageModel))

        assert saved_task is not None
        assert saved_task.title == "Reservar sala"
        assert saved_task.owner == "Alice"
        assert saved_decision is not None
        assert saved_decision.decision == "Evento presencial"
        assert saved_section is not None
        assert saved_section.section == "Cronograma"
        assert saved_message is not None
        assert saved_message.tool_calls == sample_tool_calls


def test_repository_list_users_returns_seeded_participants():
    """Retorna os participantes padrao cadastrados no banco."""
    assert repository.list_users() == ["Alice", "Bob", "Charlie"]


def test_repository_task_crud_and_case_insensitive_owner_normalization():
    """Cria, lista e atualiza tarefas retornando dicts e normalizando nomes de usuarios."""
    created = repository.create_task(
        title="Contratar coffee break",
        owner="alice",
        deadline="quarta-feira",
    )
    assert isinstance(created, dict)
    assert created["id"] >= 1
    assert created["title"] == "Contratar coffee break"
    assert created["owner"] == "Alice"  # Normalizado de 'alice' para 'Alice'
    assert created["deadline"] == "quarta-feira"
    assert created["status"] == "pendente"

    default_task = repository.create_task(title="Preparar certificados")
    assert default_task["owner"] == "a definir"
    assert default_task["deadline"] == "a definir"

    updated = repository.update_task_by_id(
        task_id=created["id"],
        status="concluida",
        owner="BOB",
        deadline="terca-feira",
    )
    assert updated is not None
    assert updated["id"] == created["id"]
    assert updated["status"] == "concluida"
    assert updated["owner"] == "Bob"
    assert updated["deadline"] == "terca-feira"

    assert repository.update_task_by_id(task_id=99999, status="concluida") is None
    assert len(repository.list_tasks()) == 2


def test_repository_decisions_and_document_sections_upsert():
    """Persiste decisoes e faz upsert em secoes do documento sem duplicar chaves."""
    d1 = repository.create_decision(
        decision="Transmissao ao vivo no YouTube",
        reason="Alcancar alunos remotos",
    )
    assert d1["id"] >= 1
    assert d1["decision"] == "Transmissao ao vivo no YouTube"
    assert d1["reason"] == "Alcancar alunos remotos"
    assert len(repository.list_decisions()) == 1

    repository.upsert_document_section("Objetivos", "Versao inicial curta.")
    repository.upsert_document_section("Objetivos", "Versao final detalhada do evento.")

    sections = repository.get_document_sections()
    assert len(sections) == 1
    assert sections["Objetivos"]["text"] == "Versao final detalhada do evento."


def test_repository_append_and_load_messages_with_safe_context_window():
    """Persiste mensagens incrementalmente e recorta janela de contexto sem quebrar pares tool_call."""
    turn_1 = [
        HumanMessage(content="[user:Alice] Crie a tarefa de divulgacao"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "register_task",
                    "args": {"title": "Divulgacao"},
                    "id": "call_div_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Tarefa #1 registrada: Divulgacao",
            tool_call_id="call_div_1",
        ),
        AIMessage(content="Registrei a tarefa de divulgacao!"),
    ]
    turn_2 = [
        HumanMessage(content="[user:Bob] Obrigado!"),
        AIMessage(content="Por nada, Bob!"),
    ]

    repository.append_messages(turn_1)
    repository.append_messages(turn_2)

    all_msgs = repository.load_messages()
    assert len(all_msgs) == 6
    assert isinstance(all_msgs[0], HumanMessage)
    assert isinstance(all_msgs[1], AIMessage) and all_msgs[1].tool_calls
    assert isinstance(all_msgs[2], ToolMessage) and all_msgs[2].tool_call_id == "call_div_1"
    assert isinstance(all_msgs[3], AIMessage)

    # Se pedirmos max_messages=4, o corte bruto pegaria [ToolMessage, AIMessage, HumanMessage, AIMessage].
    # O recorte seguro nunca deve iniciar com ToolMessage orfa!
    window = repository.load_context_messages(max_messages=4)
    assert not isinstance(window[0], ToolMessage)
    assert window[-1].content == "Por nada, Bob!"


def test_clear_project_board_resets_data_and_preserves_users():
    """Limpa tarefas, decisoes, secoes e mensagens mantendo os usuarios cadastrados."""
    repository.create_task(title="Tarefa temporaria")
    repository.create_decision(decision="Decisao temporaria")
    repository.upsert_document_section("Resumo", "Texto temporario")
    repository.append_messages([HumanMessage(content="Oi")])

    repository.clear_project_board()

    board = repository.get_project_board()
    assert board == {"tasks": [], "decisions": [], "document_sections": {}}
    assert repository.load_messages() == []
    assert repository.list_users() == ["Alice", "Bob", "Charlie"]


def test_ui_helpers_extract_tools_and_reset_database_state():
    """Valida que os helpers da UI extraem ferramentas do historico do banco e limpam a sessao."""
    from src import ui

    repository.create_task(title="Configurar projetor", owner="Charlie")
    repository.append_messages(
        [
            HumanMessage(content="[user:Charlie] Configure o projetor"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "register_task",
                        "args": {"title": "Configurar projetor"},
                        "id": "call_proj_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="Tarefa #1 registrada", tool_call_id="call_proj_1"),
            AIMessage(content="Tarefa registrada com sucesso."),
        ]
    )

    loaded = repository.load_messages()
    assert ui._extract_tools_called(loaded) == {"register_task"}

    ui._reset_session()
    assert repository.get_project_board()["tasks"] == []
    assert repository.load_messages() == []


def test_clear_project_board_executes_truncate_restart_identity_on_postgresql(monkeypatch):
    """Garante que no dialeto postgresql o clear_project_board usa TRUNCATE ... RESTART IDENTITY."""
    from unittest.mock import MagicMock

    mock_engine = MagicMock()
    mock_engine.dialect.name = "postgresql"
    mock_session = MagicMock()

    @database.contextmanager
    def fake_session():
        yield mock_session

    monkeypatch.setattr(repository, "init_db", lambda: None)
    monkeypatch.setattr(repository, "get_engine", lambda: mock_engine)
    monkeypatch.setattr(repository, "get_db_session", fake_session)

    repository.clear_project_board()

    assert mock_session.execute.call_count == 1
    executed_clause = str(mock_session.execute.call_args[0][0])
    assert "TRUNCATE TABLE tasks, decisions, document_sections, chat_messages" in executed_clause
    assert "RESTART IDENTITY CASCADE" in executed_clause


