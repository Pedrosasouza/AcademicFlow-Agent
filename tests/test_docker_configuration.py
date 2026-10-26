"""
Testes automatizados para validar a containerizacao (Dockerfile, .dockerignore e docker-compose.yml).
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerignore_excludes_sensitive_and_cache_files() -> None:
    """Garante que o .dockerignore existe e impede o envio de .env, .git e caches para a imagem."""
    dockerignore_path = PROJECT_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), "O arquivo .dockerignore deve existir na raiz do projeto."

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = {
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert ".env" in lines
    assert ".git" in lines
    assert "__pycache__/" in lines or "__pycache__" in lines
    assert ".pytest_cache/" in lines or ".pytest_cache" in lines
    assert ".venv/" in lines or ".venv" in lines


def test_dockerfile_follows_python_and_streamlit_best_practices() -> None:
    """Valida a estrutura do Dockerfile da aplicacao (imagem slim, cache de pip, usuario nao-root e Streamlit)."""
    dockerfile_path = PROJECT_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "O arquivo Dockerfile deve existir na raiz do projeto."

    content = dockerfile_path.read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in content
    assert "PYTHONDONTWRITEBYTECODE=1" in content
    assert "PYTHONUNBUFFERED=1" in content
    assert "pip install --no-cache-dir -r requirements.txt" in content
    assert "USER " in content
    assert "EXPOSE 8501" in content
    assert "src/ui.py" in content
    assert "--server.address=0.0.0.0" in content


def test_docker_compose_defines_app_service_connected_to_postgres_and_host_ollama() -> None:
    """Garante que o docker-compose.yml orquestra os servicos postgres e app com rede interna e healthcheck."""
    compose_path = PROJECT_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "O arquivo docker-compose.yml deve existir na raiz do projeto."

    content = compose_path.read_text(encoding="utf-8")

    assert "postgres:" in content
    assert "pgvector/pgvector:pg16" in content
    assert "app:" in content
    assert "build:" in content
    assert "condition: service_healthy" in content
    assert "@postgres:5432/" in content
    assert "host.docker.internal:11434" in content
    assert "host.docker.internal:host-gateway" in content
    assert "8501:8501" in content
