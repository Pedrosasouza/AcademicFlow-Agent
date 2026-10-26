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
        "alembic_version",
        "users",
        "tasks",
        "decisions",
        "document_sections",
        "chat_messages",
        "regulation_chunks",
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

    monkeypatch.setattr(repository, "get_engine", lambda: mock_engine)
    monkeypatch.setattr(repository, "get_db_session", fake_session)

    repository.clear_project_board()

    assert mock_session.execute.call_count == 1
    executed_clause = str(mock_session.execute.call_args[0][0])
    assert "TRUNCATE TABLE tasks, decisions, document_sections, chat_messages" in executed_clause
    assert "RESTART IDENTITY CASCADE" in executed_clause


def test_multi_session_isolation_across_tasks_decisions_sections_and_chat():
    """Garante isolamento multi-tenant: dados e secoes de mesmo nome nao vazam nem colidem entre sessoes."""
    t_a = repository.create_task("Reservar palco", owner="Alice", session_id="evento_a")
    t_b = repository.create_task("Contratar som", owner="Bob", session_id="evento_b")

    repository.create_decision("Entrada franca", reason="Acessibilidade", session_id="evento_a")
    repository.create_decision("Inscricao paga", reason="Custear kit", session_id="evento_b")

    # Ambas as sessoes criam uma secao com o mesmo titulo 'Cronograma' (valida UniqueConstraint composta)
    repository.upsert_document_section("Cronograma", "Cronograma do Evento A", session_id="evento_a")
    repository.upsert_document_section("Cronograma", "Cronograma do Evento B", session_id="evento_b")

    repository.append_messages([HumanMessage(content="Mensagem da sala A")], session_id="evento_a")
    repository.append_messages([HumanMessage(content="Mensagem da sala B")], session_id="evento_b")

    # Tentar atualizar tarefa da sessao A a partir da sessao B deve retornar None (isolamento de escrita)
    assert repository.update_task_by_id(t_a["id"], status="concluida", session_id="evento_b") is None
    assert repository.update_task_by_id(t_b["id"], status="concluida", session_id="evento_b") is not None

    board_a = repository.get_project_board(session_id="evento_a")
    board_b = repository.get_project_board(session_id="evento_b")

    assert len(board_a["tasks"]) == 1
    assert board_a["tasks"][0]["title"] == "Reservar palco"
    assert board_a["tasks"][0]["status"] == "pendente"
    assert board_a["decisions"][0]["decision"] == "Entrada franca"
    assert board_a["document_sections"]["Cronograma"]["text"] == "Cronograma do Evento A"

    assert len(board_b["tasks"]) == 1
    assert board_b["tasks"][0]["title"] == "Contratar som"
    assert board_b["tasks"][0]["status"] == "concluida"
    assert board_b["decisions"][0]["decision"] == "Inscricao paga"
    assert board_b["document_sections"]["Cronograma"]["text"] == "Cronograma do Evento B"

    msgs_a = repository.load_messages(session_id="evento_a")
    msgs_b = repository.load_messages(session_id="evento_b")
    assert [m.content for m in msgs_a] == ["Mensagem da sala A"]
    assert [m.content for m in msgs_b] == ["Mensagem da sala B"]


def test_list_sessions_and_selective_session_clear():
    """Lista todas as sessoes ativas e limpa apenas a sessao especificada preservando as demais."""
    repository.create_task("Tarefa Evento 1", session_id="semana_computacao")
    repository.create_decision("Decisao Evento 2", session_id="simposio_ia")

    sessions = repository.list_sessions()
    assert "default" in sessions
    assert "semana_computacao" in sessions
    assert "simposio_ia" in sessions

    repository.clear_project_board(session_id="semana_computacao")

    assert repository.get_project_board(session_id="semana_computacao")["tasks"] == []
    assert len(repository.get_project_board(session_id="simposio_ia")["decisions"]) == 1


def test_repository_index_and_vector_search_regulation_chunks():
    """Indexa trechos de editais com embeddings e ordena a busca por similaridade vetorial."""
    chunk_auditorio = repository.index_regulation_chunk(
        source="Regimento ICMC-USP",
        title="Reserva de Auditorio Exclusivo",
        content="A solicitacao do auditorio principal exige 30 dias de antecedencia no sistema.",
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    repository.index_regulation_chunk(
        source="Edital PRCEU 02/2026",
        title="Regras de Certificados",
        content="A emissao de certificados exige presenca minima de 75 por cento.",
        embedding=[0.0, 1.0, 0.0, 0.0],
    )

    assert chunk_auditorio["id"] >= 1
    assert chunk_auditorio["title"] == "Reserva de Auditorio Exclusivo"

    results = repository.search_regulation_chunks(
        query="reserva auditorio",
        query_embedding=[0.99, 0.01, 0.0, 0.0],
        top_k=2,
    )
    assert len(results) >= 1
    assert results[0]["title"] == "Reserva de Auditorio Exclusivo"
    assert results[0]["score"] > 0.8


def test_repository_executes_pgvector_cosine_operator_on_postgresql(monkeypatch):
    """Garante que no PostgreSQL com pgvector ativo a busca usa a coluna nativa vector(N) com <=> sem CAST na coluna."""
    from unittest.mock import MagicMock

    mock_engine = MagicMock()
    mock_engine.dialect.name = "postgresql"
    mock_session = MagicMock()
    mock_row = MagicMock()
    mock_row._mapping = {
        "id": 1,
        "source": "Edital USP",
        "title": "Norma de Teste",
        "content": "Conteudo do edital",
        "score": 0.95,
    }
    mock_session.execute.return_value.all.return_value = [mock_row]

    @database.contextmanager
    def fake_session():
        yield mock_session

    monkeypatch.setattr(repository, "get_engine", lambda: mock_engine)
    monkeypatch.setattr(repository, "get_db_session", fake_session)
    monkeypatch.setattr(database, "is_pgvector_enabled", lambda: True)

    results = repository.search_regulation_chunks(
        query="norma",
        query_embedding=[0.5, 0.5, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["title"] == "Norma de Teste"
    executed_sql = str(mock_session.execute.call_args[0][0])
    assert "embedding <=>" in executed_sql
    assert "embedding_text" not in executed_sql


def test_repository_operations_do_not_invoke_init_db(monkeypatch):
    """Garante que operacoes CRUD do repositorio nao reexecutam init_db() a cada chamada."""
    calls: list[int] = []
    monkeypatch.setattr(database, "init_db", lambda: calls.append(1))

    repository.create_task("Testar ciclo de vida", owner="Alice")
    repository.list_tasks()
    repository.get_project_board()
    repository.search_regulation_chunks("auditorio")

    assert calls == []


def test_compute_embedding_uses_ollama_and_supports_arbitrary_vocabulary(monkeypatch):
    """Garante que compute_embedding invoca OllamaEmbeddings em modo padrao e recupera termos como patrocinio."""
    from unittest.mock import MagicMock

    fake_embedder = MagicMock()
    fake_embedder.embed_query.return_value = [1.0, 0.5] + [0.0] * (models.EMBEDDING_DIM - 2)
    captured_kwargs: dict[str, str] = {}

    def make_embedder(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_embedder

    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL", raising=False)
    database._get_ollama_embedder.cache_clear()
    monkeypatch.setattr(database, "OllamaEmbeddings", make_embedder)

    vec = database.compute_embedding("regras de patrocinio corporativo")
    assert fake_embedder.embed_query.called
    assert captured_kwargs["model"] == "nomic-embed-text"
    assert len(vec) == models.EMBEDDING_DIM

    # Se o modelo retornar dimensao diferente de 768, deve falhar explicitamente em vez de dobrar
    fake_embedder.embed_query.return_value = [1.0, 0.5, 0.0]
    with pytest.raises(ValueError, match="retornou 3 dimensoes"):
        database.compute_embedding("vetor com dimensao errada")

    # Em modo offline/trigram, consultas para termos novos (ex.: patrocinio) devem funcionar sem dicionario fixo
    monkeypatch.setenv("EMBEDDING_PROVIDER", "trigram")
    repository.index_regulation_chunk(
        source="Norma de Parcerias USP",
        title="Diretrizes de Patrocinio e Apoio Financeiro",
        content="Contratos de patrocinio com empresas exigem aprovacao juridica e contrapartida institucional.",
    )
    matches = repository.search_regulation_chunks("como funciona o patrocinio de empresas?", top_k=1)
    assert len(matches) == 1
    assert "Patrocinio" in matches[0]["title"]
    assert matches[0]["score"] > 0.3


def test_run_migrations_stamps_legacy_db_without_dropping_data(monkeypatch, tmp_path):
    """Garante que um banco pre-existente sem alembic_version recebe stamp e preserva seus dados."""
    from sqlalchemy import text

    db_file = tmp_path / "legacy_preserve.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file.as_posix()}")
    database.reset_engine()

    engine = database.get_engine()
    models.Base.metadata.create_all(bind=engine)
    with database.get_db_session() as session:
        session.add(models.UserModel(name="Diana"))

    database.run_migrations("head")

    with database.get_db_session() as session:
        users = session.scalars(select(models.UserModel.name)).all()
    assert "Diana" in users

    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version == "0001_initial_schema"


def test_alembic_migrations_upgrade_and_downgrade_cleanly(monkeypatch, tmp_path):
    """Garante que o Alembic aplica upgrade('head') e downgrade('base') sem URLs fixas em alembic.ini."""
    from pathlib import Path
    from sqlalchemy import text

    project_root = Path(__file__).resolve().parents[1]
    alembic_ini = project_root / "alembic.ini"
    assert alembic_ini.exists(), "O arquivo alembic.ini deve existir na raiz do projeto."

    ini_content = alembic_ini.read_text(encoding="utf-8")
    assert "postgresql+psycopg://" not in ini_content

    db_file = tmp_path / "alembic_test.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file.as_posix()}")
    database.reset_engine()

    database.run_migrations("head")

    engine = database.get_engine()
    inspector = inspect(engine)
    tables_after_upgrade = set(inspector.get_table_names())
    assert {
        "alembic_version",
        "users",
        "tasks",
        "decisions",
        "document_sections",
        "chat_messages",
        "regulation_chunks",
    }.issubset(tables_after_upgrade)

    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version == "0001_initial_schema"

    database.run_migrations("base")

    inspector_after_downgrade = inspect(database.get_engine())
    remaining_tables = set(inspector_after_downgrade.get_table_names()) - {"alembic_version"}
    assert remaining_tables == set()


def test_chat_message_persists_author_and_user_id_fk_without_prefix():
    """Garante que append_messages vincula user_id (FK) e salva content sem prefixo '[user:X]'."""
    repository.append_messages(
        [
            HumanMessage(content="[user:alice] Primeira mensagem com prefixo legado"),
            HumanMessage(content="Segunda mensagem limpa", name="Bob"),
        ],
        session_id="default",
    )

    with database.get_db_session() as session:
        rows = session.scalars(
            select(models.ChatMessageModel).order_by(models.ChatMessageModel.id)
        ).all()
        assert len(rows) == 2
        assert rows[0].author == "Alice"
        assert rows[0].user_id is not None
        assert rows[0].content == "Primeira mensagem com prefixo legado"
        assert rows[1].author == "Bob"
        assert rows[1].user_id is not None
        assert rows[1].content == "Segunda mensagem limpa"

    loaded = repository.load_messages(session_id="default")
    assert loaded[0].content == "Primeira mensagem com prefixo legado"
    assert loaded[0].name == "Alice"
    assert loaded[1].content == "Segunda mensagem limpa"
    assert loaded[1].name == "Bob"


def test_compute_embedding_logs_warning_on_ollama_connection_error(monkeypatch, caplog):
    """Garante que falhas de rede no Ollama geram log warning estruturado antes do fallback."""
    from httpx import ConnectError

    def broken_embedder(**kwargs):
        raise ConnectError("Ollama indisponivel")

    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(database, "OllamaEmbeddings", broken_embedder)

    with caplog.at_level("WARNING"):
        vec = database.compute_embedding("texto para testar log de fallback")

    assert len(vec) == models.EMBEDDING_DIM
    assert "Falha ao gerar embedding via Ollama" in caplog.text


def test_upsert_document_section_recovers_from_concurrent_integrity_error(monkeypatch):
    """Garante que upsert_document_section trata IntegrityError concorrente atualizando o registro existente."""
    from sqlalchemy.exc import IntegrityError

    # Pre-insere a secao no banco real
    repository.upsert_document_section("Cronograma", "Versao Inicial", session_id="default")

    real_get_db_session = repository.get_db_session

    @database.contextmanager
    def race_condition_session():
        with real_get_db_session() as session:
            real_scalar = session.scalar
            calls = {"count": 0}

            def simulated_scalar(stmt, *args, **kwargs):
                calls["count"] += 1
                # Na primeira leitura (check de existencia), finge que ainda nao existia (simulando corrida)
                if calls["count"] == 1:
                    return None
                return real_scalar(stmt, *args, **kwargs)

            monkeypatch.setattr(session, "scalar", simulated_scalar)
            yield session

    monkeypatch.setattr(repository, "get_db_session", race_condition_session)

    result = repository.upsert_document_section(
        "Cronograma",
        "Versao Atualizada Apos Corrida",
        session_id="default",
    )
    assert result["text"] == "Versao Atualizada Apos Corrida"


def test_sessions_table_persists_empty_rooms_and_incremental_load_messages_after_id():
    """Garante que create_session persiste salas vazias em sessions e load_messages suporta after_id."""
    created = repository.create_session("workshop_quantum")
    assert created == "workshop_quantum"
    assert "workshop_quantum" in repository.list_sessions()

    repository.append_messages(
        [
            HumanMessage(content="Mensagem 1", name="Alice"),
            AIMessage(content="Resposta 1"),
        ],
        session_id="workshop_quantum",
    )
    first_batch = repository.load_messages(session_id="workshop_quantum")
    assert len(first_batch) == 2
    last_id = first_batch[-1].additional_kwargs["db_id"]

    repository.append_messages(
        [HumanMessage(content="Mensagem 2 incremental", name="Bob")],
        session_id="workshop_quantum",
    )
    delta = repository.load_messages(session_id="workshop_quantum", after_id=last_id)
    assert len(delta) == 1
    assert delta[0].content == "Mensagem 2 incremental"
    assert delta[0].name == "Bob"


def test_update_task_ignores_blank_strings_and_normalizes_accented_status():
    """Garante que strings em branco nao sobrescrevem campos existentes e status com acento e aceito."""
    created = repository.create_task(
        title="Preparar crachas",
        owner="Alice",
        deadline="2026-11-10",
    )
    updated = repository.update_task_by_id(
        task_id=created["id"],
        status="Concluída",
        owner="   ",
        deadline="",
    )
    assert updated is not None
    assert updated["status"] == "concluida"
    assert updated["owner"] == "Alice"
    assert updated["deadline"] == "2026-11-10"


def test_load_messages_normalizes_legacy_tool_calls_missing_id():
    """Garante que mensagens salvas no banco sem id em tool_calls recebem fallback consistente."""
    with database.get_db_session() as session:
        session.add(
            models.ChatMessageModel(
                session_id="default",
                role="ai",
                content="",
                tool_calls=[{"name": "consult_project_board", "args": {}}],
                tool_call_id=None,
            )
        )
        session.add(
            models.ChatMessageModel(
                session_id="default",
                role="tool",
                content="Quadro vazio",
                tool_calls=None,
                tool_call_id=None,
            )
        )

    loaded = repository.load_messages(session_id="default")
    assert len(loaded) == 2
    assert isinstance(loaded[0], AIMessage)
    assert loaded[0].tool_calls[0]["id"] == "unknown_call"
    assert isinstance(loaded[1], ToolMessage)
    assert loaded[1].tool_call_id == "unknown_call"


def test_clear_project_board_none_removes_custom_sessions():
    """Garante que clear_project_board(None) remove salas customizadas da tabela sessions."""
    repository.create_session("sala_temporaria")
    assert "sala_temporaria" in repository.list_sessions()

    repository.clear_project_board(None)
    assert repository.list_sessions() == ["default"]








