from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def clean_project_board():
    import React_Projetct as agent

    agent.project_board["tasks"].clear()
    agent.project_board["decisions"].clear()
    agent.project_board["document_sections"].clear()
    yield
    agent.project_board["tasks"].clear()
    agent.project_board["decisions"].clear()
    agent.project_board["document_sections"].clear()
