import os
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def clean_project_board(monkeypatch):
    """Isola cada teste em um banco limpo (SQLite in-memory por padrao ou TEST_DATABASE_URL)."""
    test_db_url = os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("DATABASE_URL", test_db_url)

    from src import agent, database, repository

    database.reset_engine()
    database.init_db()
    repository.clear_project_board()
    if isinstance(agent.project_board, dict):
        agent.project_board["tasks"].clear()
        agent.project_board["decisions"].clear()
        agent.project_board["document_sections"].clear()
    yield
    repository.clear_project_board()
    database.reset_engine()
