<h1 align="center">🤖 AcademicFlow-Agent</h1>

<p align="center">
  Agente colaborativo de IA com arquitetura ReAct e persistência relacional em PostgreSQL para planejamento de eventos acadêmicos.
</p>

<p align="center">
  <a href="https://github.com/Pedrosasouza/AcademicFlow-Agent/actions/workflows/tests.yml"><img src="https://github.com/Pedrosasouza/AcademicFlow-Agent/actions/workflows/tests.yml/badge.svg" alt="CI Tests" /></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-Agent_Framework-00ADD8?style=flat-square" />
  <img src="https://img.shields.io/badge/LangChain-LLM_Orchestration-1C3C3C?style=flat-square" />
  <img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat-square&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?style=flat-square&logo=sqlalchemy&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-black?style=flat-square" />
</p>

---

## 💡 Sobre o Projeto

O **AcademicFlow-Agent** é um sistema colaborativo que utiliza o padrão **ReAct (Reason + Act)** para auxiliar equipes no planejamento de eventos acadêmicos. O agente interpreta mensagens de múltiplos participantes, decide autonomamente quando acionar ferramentas e persiste o quadro do projeto e o histórico de conversas em um banco de dados relacional **PostgreSQL**.

### 🎯 Destaques do Projeto

- **Arquitetura de Agentes ReAct**: Orquestração orientada a grafos de estado (`StateGraph`) com tomada de decisão dinâmica.
- **Persistência Relacional (PostgreSQL + SQLAlchemy 2.0)**: Tarefas, decisões, seções do documento, participantes e histórico de chat armazenados no banco de dados como fonte única da verdade (*Single Source of Truth*), utilizando o padrão **Repository**.
- **Gestão Segura de Contexto do LLM**: Persistência incremental de mensagens (`append_messages`) combinada com janela deslizante de contexto (`load_context_messages`) que evita estourar o limite de tokens do modelo local e impede `ToolMessage` órfãs.
- **Interface Web & CLI Multi-Evento**: Interface interativa em Streamlit com suporte a múltiplos eventos/salas isolados por `session_id` (*multi-tenant*), múltiplos participantes, painel de estado lateral sincronizado e rastreamento de ferramentas acionadas.
- **Privacidade & Execução Local**: Integração com Ollama (`llama3.2`), permitindo inferência local sem custos de API.
- **Suíte de Testes Determinística (TDD)**: Cobertura com `pytest` validando modelos ORM, camada de repositório, ferramentas de colaboração e ciclo completo do grafo ReAct em `sqlite:///:memory:` (zero dependência externa em CI) ou contra o PostgreSQL via `TEST_DATABASE_URL`.

---

## ⚙️ Arquitetura

O sistema combina o grafo de estados **ReAct** no LangGraph com uma camada de persistência desacoplada via **Repository Pattern**:

```mermaid
flowchart LR
    A([Usuário envia mensagem]) --> B[Agente interpreta intenção]
    B --> C{Precisa de ferramenta?}
    C -->|Não| D[Responde diretamente]
    C -->|Sim| E[Executa ToolNode]
    E --> F[Repository Pattern]
    F <--> G[(PostgreSQL 16)]
    F --> B
    D --> H([Persiste turno e retorna resposta])
```

| Componente | Tecnologia | Função |
|:-----------|:-----------|:-------|
| Orquestração | LangGraph | Grafo de estados (`MessagesState`) com ciclos de decisão |
| LLM | Ollama + Llama 3.2 | Raciocínio determinístico (`temperature=0`) e geração de respostas |
| Persistência | PostgreSQL 16 + SQLAlchemy 2.0 (`psycopg` v3) | Modelagem ORM tipada (`Mapped`) e transações seguras |
| Acesso a Dados | Repository Pattern (`src/repository.py`) | Isolamento completo das queries SQL em relação ao agente e à UI |
| Ferramentas | LangChain Tools | Gestão de tarefas, decisões e seções do documento colaborativo |
| Interface | Streamlit & CLI | Interação multiusuário, visualização do quadro e badges de execução |

### Esquema do Banco de Dados (`init_db` + Alembic)

O esquema é gerenciado por migrações versionadas do **Alembic** (`run_migrations("head")`) como fonte única da verdade para o DDL na inicialização da aplicação (`init_db()` em `src/database.py`), realizando em seguida o *seed* idempotente dos participantes padrão (`Alice`, `Bob`, `Charlie`) e normas institucionais:

- `sessions`: Salas/eventos colaborativos cadastrados (`id` PK, `created_at` em `DateTime(timezone=True)`).
- `users`: Participantes cadastrados (`id`, `name`, `created_at` em `DateTime(timezone=True)`).
- `tasks`: Tarefas do planejamento isoladas por sala (`id`, `session_id` FK para `sessions.id`, `title`, `owner`, `owner_id` FK para `users.id`, `deadline`, `status` validado via `TaskStatus`, `created_at` em `DateTime(timezone=True)`).
- `decisions`: Decisões tomadas e justificativas por sala (`id`, `session_id` FK para `sessions.id`, `decision`, `reason`, `created_at` em `DateTime(timezone=True)`).
- `document_sections`: Seções do plano com unicidade composta `(session_id, section)` (`id`, `session_id` FK para `sessions.id`, `section`, `text`, `updated_at` em `DateTime(timezone=True)`).
- `chat_messages`: Histórico de mensagens (`id`, `session_id` FK para `sessions.id`, `role`, `author`, `user_id` FK para `users.id`, `content`, `tool_calls` em JSON, `tool_call_id`, `created_at` em `DateTime(timezone=True)`).
- `regulation_chunks`: Base vetorial de editais e normas institucionais (`id`, `source`, `title`, `content`, `embedding`, `created_at`) com coluna nativa `vector(768)` e índice **HNSW** (`vector_cosine_ops`) no PostgreSQL (`pgvector`), gerando embeddings via `OllamaEmbeddings` (com fallback determinístico de n-gramas quando offline/CI).

---

## 🚀 Funcionalidades

| Funcionalidade | Ferramenta (`@tool`) | Descrição |
|:---------------|:---------------------|:----------|
| 📋 Registro de tarefas | `register_task` | Cria tarefas no banco com título, responsável (normalizado), prazo e status inicial |
| 🔄 Atualização de tarefas | `update_task` | Atualiza status (ex.: `concluida`), responsável ou prazo de uma tarefa pelo `id` |
| ✅ Registro de decisões | `register_decision` | Persiste decisões importantes tomadas pelo grupo e suas justificativas |
| 📄 Documento colaborativo | `update_document_section` | Cria ou atualiza (*upsert*) seções estruturadas do plano do evento |
| 📊 Quadro do projeto | `consult_project_board` | Consulta e sintetiza o estado atual de tarefas, decisões e documentos no banco |
| 📚 Busca em editais (RAG) | `search_event_regulations` | Busca semântica em normas da universidade via `pgvector` (`<=>` + índice HNSW no PostgreSQL) ou similaridade de cosseno em memória nos testes SQLite |

---

## 🏗️ Estrutura do Projeto

```text
AcademicFlow-Agent/
├── .github/
│   └── workflows/
│       └── tests.yml                  # Pipeline de CI (Pytest em SQLite e PostgreSQL 16)
├── assets/
│   └── logo.png                       # Identidade visual exibida na interface
├── docs/
│   └── TEST_TEMPLATES.md              # Roteiros de testes manuais e de aceitação
├── migrations/
│   ├── env.py                         # Ambiente Alembic integrado ao SQLAlchemy 2.0 (Base.metadata)
│   └── versions/                      # Histórico de revisões versionadas (0001_initial_schema)
├── src/
│   ├── __init__.py
│   ├── database.py                    # Engine SQLAlchemy, sessões transacionais e init_db()
│   ├── models.py                      # Modelos ORM SQLAlchemy 2.0 (User, Task, Decision, etc.)
│   ├── repository.py                  # Camada Repository (CRUD e serialização de mensagens)
│   ├── agent.py                       # Agente ReAct principal e grafo LangGraph
│   └── ui.py                          # Interface web colaborativa em Streamlit
├── tests/
│   ├── conftest.py                    # Fixtures de isolamento (SQLite in-memory / TEST_DATABASE_URL)
│   ├── test_agent_configuration.py    # Testes de configuração, CLI e fluxo do grafo ReAct
│   ├── test_collaboration_tools.py    # Testes de integração das ferramentas com o banco
│   ├── test_database_repository.py    # Testes unitários de modelos, transações, CRUD e RAG
│   ├── test_docker_configuration.py   # Testes de infraestrutura containerizada (Dockerfile e Compose)
│   └── test_ui.py                     # Testes da interface Streamlit e tratamento de falhas
├── .dockerignore                      # Exclusão de segredos (.env) e caches no contexto de build
├── .env.example                       # Exemplo de variáveis de ambiente
├── alembic.ini                        # Configuração de migrações de banco de dados (Alembic)
├── Dockerfile                         # Imagem Python 3.11 slim com usuário não-root para a UI Streamlit
├── docker-compose.yml                 # Orquestração completa (PostgreSQL 16 + App Streamlit)
├── requirements.txt                   # Dependências de execução
├── requirements-dev.txt               # Dependências de desenvolvimento e testes
└── README.md
```

---

## 🛠️ Como Executar

**Pré-requisitos:** Python 3.10+, [Docker Compose](https://docs.docker.com/compose/) (ou PostgreSQL local) e [Ollama](https://ollama.com).

```bash
# 1. Clone o repositório
git clone https://github.com/Pedrosasouza/AcademicFlow-Agent.git
cd AcademicFlow-Agent

# 2. Configure as variáveis de ambiente
cp .env.example .env

# 3. Crie e ative um ambiente virtual Python limpo e instale as dependências
python -m venv .venv
# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# No Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt

# 4. Suba o banco de dados PostgreSQL (escolha UMA das opções abaixo):

# Opção 4A — Via Docker Compose (recomendado, já inclui PostgreSQL 16 + pgvector):
docker compose up -d postgres

# Opção 4B — PostgreSQL instalado localmente no Windows (sem Docker):
# Inicie o serviço pelo PowerShell (como Administrador) e crie o banco se ainda não existir:
net start postgresql-x64-18
createdb -U postgres academic_flow

# 5. Baixe os modelos necessários no Ollama (chat + embeddings) e garanta que o servidor esteja ativo
ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve
```

> **Nota sobre migrações e *seed* inicial:** Ao iniciar a aplicação (via Streamlit ou CLI), `init_db()` executa automaticamente as migrações do **Alembic** (`alembic upgrade head`), ativa a extensão `pgvector` (quando disponível) e popula de forma idempotente os participantes (`Alice`, `Bob`, `Charlie`) e as normas institucionais iniciais. Também é possível aplicar as migrações manualmente com `alembic upgrade head`.

### Variáveis de Ambiente (`.env.example`)

| Variável | Valor Padrão | Descrição |
|:---------|:-------------|:----------|
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/academic_flow` | String de conexão SQLAlchemy 2.0 |
| `OLLAMA_MODEL` | `llama3.2` | Modelo conversacional utilizado no Ollama |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Modelo de embeddings densos (768d) utilizado no RAG |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint da API local do Ollama |

### Execução Rápida com Docker Compose (Banco + Interface Web)

Com o [Ollama](https://ollama.com) ativo na máquina hospedeira e os modelos baixados (`ollama pull llama3.2 && ollama pull nomic-embed-text && ollama serve`), você pode subir o **PostgreSQL 16 (`pgvector`)** e a **interface Streamlit** containerizada com um único comando:

```bash
docker compose up --build -d
```

Acesse a interface web em **`http://localhost:8501`**.

### Execução Local (Python no Host)

### Opção A — Interface Web (Streamlit)
```bash
streamlit run src/ui.py
```

### Opção B — Interface de Linha de Comando (CLI)
```bash
python src/agent.py
```

---

## 🧪 Testes Automatizados

A suíte automatizada (`pytest`) utiliza por padrão um banco **SQLite em memória** (`sqlite:///:memory:` com `StaticPool`) e mocks para o LLM, permitindo validar todas as camadas sem precisar de Docker ou Ollama ativos:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Caso deseje executar a mesma suíte contra uma instância **PostgreSQL real**, defina `TEST_DATABASE_URL`:

```bash
# PowerShell
$env:TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/academic_flow"
pytest tests/ -v
```

Os roteiros de testes manuais e de aceitação estão documentados em [`docs/TEST_TEMPLATES.md`](docs/TEST_TEMPLATES.md). A cada *push* ou *Pull Request*, o workflow [`.github/workflows/tests.yml`](.github/workflows/tests.yml) executa automaticamente a suíte tanto em `sqlite:///:memory:` quanto em um container de serviço `postgres:16-alpine`.

---

## ⚖️ Decisões Arquiteturais e Limitações Conhecidas

- **Simulação Multiusuário sem Autenticação:** O seletor de participante na barra lateral do Streamlit (`Alice`, `Bob`, `Charlie`) destina-se à demonstração acadêmica de fluxos colaborativos em grupo na disciplina *SSC0528 - Sistemas Colaborativos*. Não há camada de autenticação por senha/OAuth/JWT; a autoria selecionada é vinculada via chave estrangeira (`users.id`) nas tabelas `tasks` e `chat_messages`.
- **Arquitetura de RAG e Fallbacks de Teste (Geração de Embeddings vs. Busca Vetorial):** O módulo de consulta a editais (`regulation_chunks`) opera em dois níveis independentes para garantir performance em produção e zero dependência externa nos testes:
  1. *Geração do vetor (`compute_embedding`, 768d):* Em produção (`EMBEDDING_PROVIDER=ollama`), utiliza embeddings densos via `OllamaEmbeddings` (`nomic-embed-text`). Caso `OLLAMA_EMBEDDING_MODEL` seja alterado, o modelo deve produzir exatamente 768 dimensões. Apenas quando o Ollama está offline ou durante testes automatizados (`EMBEDDING_PROVIDER=trigram`), o sistema recorre ao fallback determinístico de projeção de n-gramas (`compute_trigram_embedding`).
  2. *Execução da busca (`search_regulation_chunks`):* No PostgreSQL com a extensão `pgvector` ativa, a busca ocorre diretamente no banco via operador `<=>` (distância de cosseno) acelerado pelo índice **HNSW** (`ix_regulation_chunks_embedding_hnsw`). O cálculo de similaridade de cosseno em memória (`_cosine_similarity` com *full scan*) é utilizado exclusivamente como fallback para testes em `sqlite:///:memory:` (que não possui a extensão `pgvector`).
- **Migrações Versionadas (Alembic):** Toda criação e evolução de schema ocorre via `run_migrations("head")` (`migrations/versions/`). Caso um banco legado sem tabela `alembic_version` seja detectado, o sistema aplica `alembic stamp` preservando os dados existentes.

---

## 📚 Aprendizados

- Construção de **agentes autônomos** com ciclos de raciocínio e ação (ReAct)
- Orquestração de fluxos conversacionais com **grafos de estado** (LangGraph)
- Persistência relacional com **SQLAlchemy 2.0 (`Mapped`)**, **PostgreSQL** e **Repository Pattern**
- Desenvolvimento orientado a testes (**TDD**) com isolamento transacional e mocks determinísticos

---

## 📝 Licença

Este projeto é de uso acadêmico e pessoal.

---

<p align="center">
  Desenvolvido por <a href="https://github.com/Pedrosasouza"><strong>Pedro Santos Souza</strong></a> · USP São Carlos
</p>
