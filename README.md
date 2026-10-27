# AcademicFlow-Agent

Agente colaborativo para planejamento de eventos acadêmicos construído com o padrão ReAct em LangGraph, persistência relacional em PostgreSQL e busca semântica em editais via `pgvector`.

[![CI Tests](https://github.com/Pedrosasouza/AcademicFlow-Agent/actions/workflows/tests.yml/badge.svg)](https://github.com/Pedrosasouza/AcademicFlow-Agent/actions/workflows/tests.yml)

---

## Visão Geral

O planejamento de eventos acadêmicos (semanas de curso, simpósios e workshops) em grupo costuma espalhar informações em vários lugares: tarefas combinadas no chat são esquecidas, decisões ficam sem registro do motivo e dúvidas sobre normas da universidade (prazos de reserva de auditório, limite de gastos com *coffee break* ou regras de certificados) exigem consulta manual a editais.

O **AcademicFlow-Agent** atua como um assistente compartilhado na conversa da comissão organizadora. A partir das mensagens dos participantes, o agente decide quando responder diretamente e quando acionar ferramentas para registrar tarefas, salvar decisões, atualizar seções do plano do evento no PostgreSQL ou consultar trechos de regulamentos universitários indexados com embeddings vetoriais.

---

## Funcionalidades

O agente expõe 6 ferramentas (`@tool` em `src/agent.py`) acopladas ao banco de dados através da camada de repositório:

| Ferramenta | Função |
|:-----------|:-------|
| `register_task` | Registra uma tarefa com título, responsável (informado ou inferido do usuário ativo no turno), prazo e status inicial `pendente`. |
| `update_task` | Atualiza status (`pendente`, `em_andamento`, `concluida`, `cancelada`), responsável ou prazo de uma tarefa existente pelo ID. |
| `register_decision` | Salva uma decisão tomada pelo grupo junto com a justificativa. |
| `update_document_section` | Cria ou atualiza (*upsert*) uma seção do documento de planejamento do evento. |
| `consult_project_board` | Retorna um resumo das tarefas, decisões e seções do documento da sala ativa. |
| `search_event_regulations` | Faz busca semântica nas normas e editais da universidade usando distância de cosseno no `pgvector`. |

Além das ferramentas do agente, a aplicação conta com:
- **Organização de dados por sala/evento (`session_id`)**: permite criar e alternar entre diferentes eventos (ex.: `default`, `semana_computacao`, `simposio_ia`), mantendo tarefas, documentos e histórico de chat organizados separadamente de forma lógica (sem atuar como barreira de segurança ou autenticação).
- **Sincronização incremental na interface web**: a UI em Streamlit atualiza o quadro lateral e o chat a cada 5 segundos (`@st.fragment`), buscando apenas novas mensagens no banco (`after_id`).
- **Indexação de editais pela interface**: barra lateral com formulário para adicionar novas regras institucionais na tabela vetorial em tempo de execução.
- **Janela de contexto segura**: `load_context_messages` envia apenas as mensagens mais recentes ao LLM e garante que o recorte nunca inicie com uma `ToolMessage` órfã.

---

## Arquitetura

As interfaces (Streamlit em `src/ui.py` e CLI em `src/agent.py`) chamam a função `run_turn`, que carrega o histórico recente da sessão, executa o ciclo ReAct no LangGraph e persiste as novas mensagens geradas no turno. Nenhuma query SQL fica exposta no agente ou na UI; todo o acesso a dados passa por `src/repository.py`.

```text
          Interface Streamlit (src/ui.py)  /  CLI (src/agent.py)
                                    |
                                    |  run_turn(user_input, session_id, author)
                                    v
             +----------------------------------------------+
             |           Grafo ReAct (LangGraph)            |
             |                                              |
             |   [Nó agent] <--------------> ChatOllama     |
             |     |    ^                    (llama3.2)     |
             |     v    |                                   |
             |   [ToolNode]                                 |
             +-----|----------------------------------------+
                   |
     +-------------+--------------------+-------------------+
     |                                  |                   |
     v                                  v                   v
Tarefas e Decisões             Seções e Quadro          Busca RAG
(register_task,                (update_document_section, (search_event_regulations)
 update_task,                   consult_project_board)      |
 register_decision)                     |                   v
     |                                  |            OllamaEmbeddings
     |                                  |         (nomic-embed-text, 768d)
     +------------------+---------------+-------------------+
                        |
                        v
            Repositório (src/repository.py)
            [SQLAlchemy 2.0 + Context Window]
                        |
                        v
              PostgreSQL 16 + pgvector
    (sessions, users, tasks, decisions, document_sections,
     chat_messages, regulation_chunks [vector(768) + HNSW])
```

### Esquema do Banco de Dados

Na inicialização da aplicação (`init_db()` em `src/database.py`), o sistema executa as migrações do Alembic (`run_migrations("head")`), ativa a extensão `vector` quando conectado ao PostgreSQL e insere de forma idempotente os usuários padrão (`Alice`, `Bob`, `Charlie`) e 4 normas institucionais iniciais:

- `sessions`: salas/eventos colaborativos (`id` PK, `created_at`).
- `users`: participantes cadastrados (`id` PK, `name` único, `created_at`).
- `tasks`: tarefas isoladas por sala (`id` PK, `session_id` FK, `title`, `owner`, `owner_id` FK, `deadline`, `status`, `created_at`).
- `decisions`: decisões e justificativas por sala (`id` PK, `session_id` FK, `decision`, `reason`, `created_at`).
- `document_sections`: seções do plano com restrição única composta `(session_id, section)` (`id` PK, `session_id` FK, `section`, `text`, `updated_at`).
- `chat_messages`: histórico de mensagens (`id` PK, `session_id` FK, `role`, `author`, `user_id` FK, `content`, `tool_calls` JSON, `tool_call_id`, `created_at`).
- `regulation_chunks`: trechos de editais e normas (`id` PK, `source`, `title`, `content`, `embedding` `vector(768)`, `created_at`) com índice HNSW (`vector_cosine_ops`).

---

## Tecnologias

- **Python 3.10+**: tipagem estática nas funções, modelos ORM e validadores de ferramentas.
- **LangGraph & LangChain**: construção do `StateGraph` (`MessagesState`, `ToolNode`, `tools_condition`) e integração com ferramentas.
- **Ollama**: execução local do modelo conversacional (`llama3.2` com `temperature=0`) e geração de embeddings de 768 dimensões (`nomic-embed-text`).
- **PostgreSQL 16 & pgvector**: banco relacional e busca vetorial com operador de distância de cosseno (`<=>`) e índice HNSW.
- **SQLAlchemy 2.0 (`psycopg` v3) & Alembic**: mapeamento ORM declarativo (`DeclarativeBase`, `Mapped`) e controle de versão do schema.
- **Streamlit**: interface web multiusuário com barra lateral de acompanhamento do projeto.
- **Pytest**: suíte de testes automatizados rodando em SQLite em memória ou PostgreSQL.

---

## Estrutura do Projeto

```text
AcademicFlow-Agent/
├── .github/workflows/tests.yml        # CI executando pytest em SQLite e PostgreSQL 16
├── assets/logo.png                    # Logo exibida na interface Streamlit
├── benchmarks/
│   ├── evaluate_rag.py                # Script reproduzivel de avaliacao de busca vetorial
│   ├── rag_benchmark_dataset.json     # Dataset versionado com perguntas e ground truth
│   └── results/                       # Metricas detalhadas exportadas em JSON
├── docs/
│   ├── RAG_EVALUATION.md              # Relatorio quantitativo detalhado da avaliacao do RAG
│   └── TEST_TEMPLATES.md              # Roteiros para testes manuais na interface
├── migrations/                        # Ambiente e revisões de schema do Alembic
├── src/
│   ├── database.py                    # Engine SQLAlchemy, sessões, migrações e embeddings
│   ├── models.py                      # Modelos ORM SQLAlchemy 2.0
│   ├── repository.py                  # Operações CRUD, busca vetorial e histórico de mensagens
│   ├── agent.py                       # Grafo ReAct, definição das tools e loop CLI
│   └── ui.py                          # Interface web em Streamlit
├── tests/
│   ├── conftest.py                    # Isolamento de banco para testes
│   ├── test_agent_configuration.py    # Testes do grafo ReAct e CLI
│   ├── test_collaboration_tools.py    # Testes das ferramentas de colaboração e RAG
│   ├── test_database_repository.py    # Testes de ORM, repositório, Alembic e pgvector
│   ├── test_docker_configuration.py   # Validação de Dockerfile e docker-compose.yml
│   ├── test_rag_benchmark.py          # Testes unitarios da suite de benchmark do RAG
│   ├── test_turn_recovery_and_resilience.py # Testes de falha pos-tool, recuperacao e idempotencia
│   └── test_ui.py                     # Testes da interface Streamlit
├── .env.example                       # Exemplo de variáveis de ambiente
├── alembic.ini                        # Configuração do Alembic
├── Dockerfile                         # Imagem Python 3.11-slim da aplicação
├── docker-compose.yml                 # Containers PostgreSQL 16 (pgvector) e Streamlit
├── requirements.txt                   # Dependências de execução
└── requirements-dev.txt               # Dependências de teste
```

---

## Como Executar

**Pré-requisitos:** Python 3.10+, [Docker Compose](https://docs.docker.com/compose/) (ou PostgreSQL local) e [Ollama](https://ollama.com).

```bash
# 1. Clonar o repositório e configurar o arquivo .env
git clone https://github.com/Pedrosasouza/AcademicFlow-Agent.git
cd AcademicFlow-Agent
cp .env.example .env

# 2. Criar ambiente virtual e instalar dependências
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt

# 3. Subir o banco PostgreSQL 16 com pgvector via Docker
docker compose up -d postgres

# 4. Baixar os modelos no Ollama e iniciar o servidor
ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve
```

### Inicializando a aplicação

Com o banco e o Ollama ativos, escolha uma das formas de uso:

```bash
# Interface Web (Streamlit) em http://localhost:8501
streamlit run src/ui.py

# Interface de Linha de Comando (CLI)
python src/agent.py
```

Também é possível rodar tanto o PostgreSQL quanto o container da aplicação Streamlit juntos via Docker Compose (conectando ao Ollama da máquina hospedeira):

```bash
docker compose up --build -d
```

### Variáveis de Ambiente (`.env.example`)

| Variável | Padrão | Descrição |
|:---------|:-------|:----------|
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/academic_flow` | URL de conexão do banco de dados |
| `OLLAMA_MODEL` | `llama3.2` | Modelo usado pelo agente para chat e chamada de ferramentas |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Modelo usado para gerar vetores de 768 dimensões |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endereço da API do Ollama |
| `EMBEDDING_PROVIDER` | `ollama` | Define `ollama` para uso normal ou `trigram` para execução offline/testes |

---

## Exemplo de Uso

Interação via CLI (`python src/agent.py`) mostrando uma consulta a normas da universidade seguida do registro de uma tarefa e de uma decisão:

```text
=== React Agent Colaborativo: Planejamento de Evento ===
Digite 'exit' ou 'quit' para sair.

Grupo: Qual a antecedência mínima para reservar o auditório principal e qual o limite do orçamento para coffee break?
  [chamando ferramenta: search_event_regulations]
  [resultado da ferramenta: Normas e editais encontrados:
- [Regimento de Espacos ICMC-USP | Reserva de Auditorio e Salas] A solicitacao de reserva do auditorio principal para eventos academicos exige antecedencia minima de 30 dias corridos no sistema institucional e aprovacao da Comissao de Cultura e Extensao.
- [Edital PRCEU-USP 01/2026 | Orcamento e Prestacao de Contas de Coffee Break] Despesas com coffee break e alimentacao sao limitadas a 20% do valor total aprovado no edital, exigindo emissao de nota fiscal com CNPJ da universidade e lista de presenca assinada.]
Agente: Segundo as normas cadastradas, a reserva do auditório principal exige pelo menos 30 dias corridos de antecedência no sistema institucional e aprovação da Comissão de Cultura e Extensão. Já as despesas com coffee break são limitadas a 20% do valor aprovado no edital, com emissão de nota fiscal no CNPJ da universidade e lista de presença assinada. Querem que eu registre uma tarefa para providenciar a reserva?

Grupo: Sim, crie uma tarefa para reservar o auditório até sexta-feira com a Alice como responsável.
  [chamando ferramenta: register_task]
  [resultado da ferramenta: Tarefa #1 registrada: Reservar o auditório | responsavel: Alice | prazo: sexta-feira.]
Agente: Tarefa #1 registrada para Alice com prazo até sexta-feira.

Grupo: Registre também a decisão de fazer o evento presencial no auditório principal porque comporta mais participantes.
  [chamando ferramenta: register_decision]
  [resultado da ferramenta: Decisao #1 registrada: Fazer o evento presencial no auditório principal. Motivo: Comporta mais participantes]
Agente: Decisão #1 registrada no quadro do projeto.
```

Na interface Streamlit (`streamlit run src/ui.py`), essas mesmas chamadas atualizam os contadores na barra lateral (`Tarefas`, `Decisões`, `Seções`), listam os itens criados e exibem indicadores com os nomes das ferramentas acionadas na sessão.

---

## Testes

Os testes automatizados usam por padrão um banco SQLite em memória (`sqlite:///:memory:` com `StaticPool`) e mocks para o LLM, portanto rodam sem depender de Docker ou do Ollama abertos:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Para executar a suíte contra uma instância PostgreSQL real, basta definir `TEST_DATABASE_URL`:

```powershell
$env:TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/academic_flow"
pytest tests/ -v
```

O que é coberto em `tests/`:
- `test_database_repository.py`: migrações Alembic (`upgrade`, `downgrade` e `stamp` em bancos legados), rollback de sessão em caso de exceção, CRUD completo, isolamento entre sessões (`session_id`), recorte da janela de contexto sem `ToolMessage` órfã, geração de embeddings com `OllamaEmbeddings` e busca vetorial via `pgvector`.
- `test_collaboration_tools.py`: execução das 6 tools sobre o banco, atribuição automática do usuário ativo via `ContextVar`, validação de status/prazos e normalização de argumentos aninhados gerados pelo modelo local.
- `test_agent_configuration.py`: compilação do grafo no LangGraph, ciclo completo de chamada de ferramenta, resposta direta e tratamento de erros de conexão na CLI.
- `test_ui.py`: sincronização de estado da interface Streamlit, criação de salas, cadastro de normas no painel lateral e mensagens de erro quando o banco ou o Ollama estão inacessíveis.
- `test_docker_configuration.py`: verificação de boas práticas no `Dockerfile`, `.dockerignore` e `docker-compose.yml`.
- `test_rag_benchmark.py`: cálculo determinístico de métricas de recuperação (Precision@k, Recall@k, MRR), detecção de falso positivo para perguntas fora de escopo e geração do relatório técnico.
- `test_turn_recovery_and_resilience.py`: resiliência do turno quando o LLM falha após a execução de ferramentas, persistência de histórico parcial com aviso de recuperação e idempotência contra duplicatas.

---

## Avaliação Quantitativa do RAG

Para evidenciar empiricamente a eficácia da recuperação vetorial sobre as normas institucionais, o projeto conta com um pipeline de benchmark reproduzível em `benchmarks/evaluate_rag.py` executado sobre um dataset versionado (`benchmarks/rag_benchmark_dataset.json`):

```bash
# Executa o benchmark e gera os relatórios em Markdown e JSON
python benchmarks/evaluate_rag.py --output-markdown docs/RAG_EVALUATION.md --output-json benchmarks/results/rag_evaluation_report.json
```

### Resultados Empíricos Observados (PostgreSQL 16 + pgvector + nomic-embed-text)

| Métrica | Resultado | O que avalia |
|:--------|:----------|:-------------|
| **Precision@1** | 91.67% | Proporção de consultas em que o primeiro resultado retornado é o documento relevante esperado |
| **Precision@3** | 30.55% | Proporção de itens relevantes no top-3 retornado (cada pergunta possui 1 trecho relevante no banco) |
| **Recall@1** | 91.67% | Cobertura do documento relevante logo no primeiro resultado |
| **Recall@3** | 91.67% | Cobertura do documento relevante considerando até 3 candidatos |
| **MRR (Mean Reciprocal Rank)** | 0.9167 | Média de 1/rank da primeira norma relevante encontrada nas consultas no escopo |
| **Latência (Mediana / P50)** | 189.87 ms | Tempo mediano por consulta semântica (latência média geral de 427.86 ms incluindo inicialização) |
| **Respostas sem suporte (sem filtro)** | 100.00% | Consultas fora de escopo (ex.: restaurante universitário) recebem candidatos espúrios se nenhum limiar for aplicado |
| **Respostas sem suporte (com limiar >= 0.68)** | 33.33% | Rejeição de 66.7% das perguntas fora de escopo (rejeição de 100% ao calibrar limiar para 0.70) |

O relatório detalhado com a discriminação por consulta e análise de calibração de limiar está documentado em [`docs/RAG_EVALUATION.md`](docs/RAG_EVALUATION.md).

---

## Limitações Conhecidas e Decisões de Projeto

- **Modelo multiusuário e limites do isolamento por sessão (`session_id`)**:
  - *Organização de dados vs. segurança de acesso*: o particionamento por `session_id` e o seletor de participantes (`Alice`, `Bob`, `Charlie`) funcionam como um mecanismo de **organização lógica dos dados** (para gerenciar múltiplos eventos de forma isolada no mesmo banco) e para viabilizar simulações de dinâmica de grupo na disciplina acadêmica (*SSC0528 - Sistemas Colaborativos*).
  - *Ausência de autenticação e autorização*: o sistema **não implementa controle de acesso (RBAC), autenticação de identidade, senhas ou tokens criptográficos (JWT/OAuth)**. Na interface web ou na CLI, qualquer participante pode alternar livremente entre salas ou selecionar qualquer identidade para envio de mensagens.
  - *Adaptação para produção*: em um ambiente produtivo multiusuário real, seria mandatório desacoplar a identidade da interface através de um provedor de autenticação externo (ex.: OAuth 2.0 / OpenID Connect ou JWT assinado) e validar autorizações de leitura e escrita por usuário antes de executar queries vinculadas a um `session_id`.
- **Fallbacks de RAG para ambiente de teste**:
  1. *Geração de embeddings (`compute_embedding`)*: em uso normal (`EMBEDDING_PROVIDER=ollama`), os vetores de 768 dimensões são gerados pelo modelo `nomic-embed-text` no Ollama. Caso o Ollama esteja desligado ou durante a execução do `pytest` (`EMBEDDING_PROVIDER=trigram`), o código usa uma projeção determinística baseada em n-gramas para permitir testes offline.
  2. *Consulta vetorial (`search_regulation_chunks`)*: no PostgreSQL com `pgvector`, a ordenação ocorre direto no banco via operador `<=>` e índice HNSW. No SQLite em memória (usado nos testes unitários), a similaridade de cosseno é calculada em Python sobre os registros de teste.
- **Ingestão de editais em texto puro**: novas normas são cadastradas como trechos de texto (via *seed* inicial ou pelo formulário na barra lateral). O projeto ainda não possui extração e *chunking* automático a partir de arquivos PDF.

---

## Autor

Desenvolvido por [Pedro Santos Souza](https://github.com/Pedrosasouza) — USP São Carlos.
