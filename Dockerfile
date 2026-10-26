FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Instala dependencias primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia apenas o codigo-fonte, migracoes e ativos visuais necessarios em producao
COPY alembic.ini ./
COPY migrations/ ./migrations/
COPY src/ ./src/
COPY assets/ ./assets/

# Cria usuario nao-root por seguranca
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "src/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]
