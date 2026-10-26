<!-- Generated from RULES.md for Antigravity / AI Pair Programming -->
# Regras do projeto: AcademicFlow-Agent

## Contexto

Agente de IA para planejamento acadêmico. Usa o padrão ReAct, orquestrado com LangGraph e LangChain. O modelo (Llama 3.2) roda localmente com Ollama. Os testes são feitos com Pytest.

**Persistência com banco de dados**:
- **Banco em produção/local**: PostgreSQL 16 via Docker Compose (`docker-compose.yml`), conectado com SQLAlchemy 2.0 e driver `psycopg` (v3).
- **Banco em testes automatizados**: SQLite em memória (`sqlite:///:memory:` com `StaticPool`), garantindo testes rápidos, determinísticos e sem dependência de container rodando.
- **Objetivo**: Substituir o estado em memória (`project_board` e `st.session_state`) por persistência relacional com histórico multiusuário em tempo real (tarefas, decisões, seções do plano e mensagens de chat).

Este é um projeto de portfólio para candidaturas a estágio em Engenharia de IA. Eu preciso entender e conseguir explicar cada parte do código.

---

## Como trabalhar

1. **Leitura prévia**: Antes de mudar qualquer coisa, leia a estrutura do repositório, o README e os testes existentes. Não assuma pastas, arquivos ou dependências que você não viu.
2. **Plano prévio aprovado**: A implementação segue o plano em 6 fases:
   - *Fase 1*: Infraestrutura e Dependências (`requirements.txt`, `docker-compose.yml`, `.env.example`, `.gitignore`).
   - *Fase 2*: Camada de Banco (`src/database.py`, `src/models.py`).
   - *Fase 3*: Camada de Repositório (`src/repository.py`).
   - *Fase 4*: Integração com o Agente ReAct (`src/agent.py`).
   - *Fase 5*: Integração com a UI Streamlit (`src/ui.py`).
   - *Fase 6*: Testes Automatizados e Documentação (`tests/conftest.py`, `tests/test_database_repository.py`, `README.md`).
3. **Passos incrementais**: Faça mudanças pequenas e focadas. Conclua e valide uma fase antes de passar para a próxima.
4. **Didática e clareza**: Explique em linguagem simples o que fez e por quê (ex.: por que usar o padrão Repository, por que SQLAlchemy 2.0 com `Mapped`, como funciona a serialização das mensagens do LangChain). Eu estou aprendendo e vou defender isso em entrevista.
5. **Dependências**: As novas dependências acordadas são `sqlalchemy>=2.0`, `psycopg[binary]>=3.1` e `python-dotenv`. Não adicione outras bibliotecas sem autorização prévia.
6. **Comportamento do agente**: Mantenha o comportamento das ferramentas e do fluxo ReAct idêntico ao atual, adaptando apenas a fonte de dados para o repositório.

---

## Código

- **Linguagem & Tipagem**: Python 3.10+, com type hints estritos nas funções novas e modelos ORM.
- **Nomenclatura**: Nomes de variáveis, funções, classes e módulos em inglês (`UserModel`, `TaskRepository`, `get_project_board`), sem abreviações obscuras. Comentários e docstrings explicativos em português.
- **Qualidade**: Código limpo, sem código morto e sem comentários redundantes.
- **Configurações**: Todas as configurações via variáveis de ambiente (`DATABASE_URL`, `OLLAMA_MODEL`, `OLLAMA_BASE_URL`) carregadas com `python-dotenv`. Nada de credencial ou URL fixa no código.
- **Segurança**: Nunca commitar segredos ou arquivos `.env`. Manter `.env.example` atualizado com exemplos e checar `.gitignore`.
- **Tratamento de erros**: Tratar explicitamente exceções de conexão com o banco (`OperationalError`), falhas do Ollama e dados inválidos, exibindo mensagens claras e acionáveis.

---

## Banco de dados & Arquitetura

- **Padrão Repository**: Acesso a dados centralizado exclusivamente em `src/repository.py`. Nem o agente (`agent.py`) nem a interface (`ui.py`) devem conter queries SQL diretas ou sessões ORM dispersas.
- **Modelagem ORM**: Uso estrito da sintaxe moderna do SQLAlchemy 2.0 (`DeclarativeBase`, `Mapped[...]`, `mapped_column(...)`).
- **Entidades mapeadas**:
  - `UserModel`: Participantes do projeto (seed automático: Alice, Bob, Charlie).
  - `TaskModel`: Tarefas com título, responsável, prazo e status.
  - `DecisionModel`: Decisões registradas e justificativas.
  - `DocumentSectionModel`: Seções estruturadas do documento de planejamento.
  - `ChatMessageModel`: Histórico de mensagens do chat (papel, conteúdo, metadados de tool calls).
- **Reprodutibilidade**: Esquema criado programaticamente via `init_db()` (`Base.metadata.create_all`) de forma idempotente.
- **Segurança**: Consultas parametrizadas garantidas via ORM/SQLAlchemy Core, sem concatenação de strings em SQL.
- **Isolamento**: Testes rodam em banco SQLite em memória isolado; nunca dependem de um banco populado nem afetam dados reais.

---

## Testes

- **Cobertura obrigatória**: Todo método do repositório e alteração nas tools deve possuir cobertura em Pytest.
- **Zero dependência externa**: Os testes unitários e de integração de ferramentas **não podem** depender do Docker/PostgreSQL nem do Ollama rodando. Utilizar `sqlite:///:memory:` e mocks do LLM.
- **Validação real**: Execute `pytest` antes de declarar qualquer etapa como concluída e apresente a saída do terminal.
- **Integridade**: Se um teste quebrar, investigue a causa raiz e corrija o código. É proibido enfraquecer asserções ou remover testes para forçar sucesso.

---

## Commits

Conventional Commits, em português: `tipo(escopo): resumo no imperativo`.

- **Tipos**: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.
- **Formato do título**: Até 70 caracteres, sem ponto final, sem emoji, sem marketing.
- **Escopo atômico**: Um commit por fase ou mudança lógica (ex.: `feat(infra)`, `feat(db)`, `refactor(agent)`, `test(db)`).
- **Corpo do commit**: Justifique as decisões arquiteturais (ex.: escolha de SQLAlchemy 2.0 e psycopg3 para compatibilidade e tipagem).
- **Sem push direto**: **Nunca faça push.** Apresente o `git log` para revisão prévia.

### Exemplos:
```text
feat(infra): adiciona docker-compose e configuracao de ambiente para postgres
feat(db): implementa modelos orm e inicializacao do banco com sqlalchemy 2
feat(repository): implementa operacoes crud para tarefas, decisoes e chat
refactor(agent): integra ferramentas react com a camada de repositorio
refactor(ui): conecta interface streamlit ao banco de dados relacional
test(db): cobre persistencia do repositorio e isolamento com sqlite in-memory
docs(readme): documenta execucao com postgres via docker compose
```

---

## README

Ao finalizar ou alterar a infraestrutura, atualize o README com:
- Como subir o banco localmente com `docker compose up -d`.
- Variáveis de ambiente no `.env.example`.
- Como executar a aplicação e os testes sem dependências externas.
- Instruções diretas e realistas sobre o que de fato está implementado e validado.

---

## Honestidade

- Não invente funcionalidade, resultado de teste ou comando que você não executou.
- Se não tiver certeza, diga. Se uma abordagem tiver risco, avise antes de seguir.
- Se eu pedir algo que você ache uma má ideia, diga por quê e proponha alternativa.
