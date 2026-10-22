import pytest
from sqlalchemy import inspect, select

from src import database, models


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
