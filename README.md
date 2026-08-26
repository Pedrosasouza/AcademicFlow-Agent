<h1 align="center">🤖 AcademicFlow-Agent</h1>

<p align="center">
  Agente colaborativo de IA com arquitetura ReAct para planejamento de eventos acadêmicos.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-Agent_Framework-00ADD8?style=flat-square" />
  <img src="https://img.shields.io/badge/LangChain-LLM_Orchestration-1C3C3C?style=flat-square" />
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-black?style=flat-square" />
  <img src="https://img.shields.io/badge/Llama_3.2-Meta-0467DF?style=flat-square" />
</p>

---

## 💡 Sobre o Projeto

O **AcademicFlow-Agent** é um sistema inteligente que utiliza o padrão **ReAct (Reason + Act)** para auxiliar equipes no planejamento de eventos acadêmicos. O agente analisa mensagens dos usuários, decide autonomamente quando usar ferramentas e mantém o estado do planejamento atualizado em tempo real.

### Por que este projeto é relevante?

- Demonstra domínio de **arquitetura de agentes de IA** com orquestração via grafo de estados
- Implementa **Tool Calling** — o agente decide dinamicamente quais ferramentas acionar
- Utiliza **LLM local** (Ollama + Llama 3.2) — sem dependência de APIs pagas
- Inclui **testes automatizados** para validar configuração e ferramentas

---

## ⚙️ Arquitetura

O sistema foi construído como um **State Graph** no LangGraph, seguindo o padrão ReAct:

```mermaid
flowchart LR
    A([Usuário envia mensagem]) --> B[Agente interpreta intenção]
    B --> C{Precisa de ferramenta?}
    C -->|Não| D[Responde diretamente]
    C -->|Sim| E[Executa ferramenta]
    E --> F[Atualiza estado do projeto]
    F --> B
    D --> G([Retorna resposta contextualizada])
```

| Componente | Tecnologia | Função |
|:-----------|:-----------|:-------|
| Orquestração | LangGraph | Grafo de estados com ciclos de decisão |
| LLM | Ollama + Llama 3.2 | Raciocínio e geração de respostas |
| Ferramentas | LangChain Tools | Registro de tarefas, decisões e documentos |
| Interface | Python CLI / UI | Interação com o usuário |

---

## 🚀 Funcionalidades

| Funcionalidade | Descrição |
|:---------------|:----------|
| 📋 Registro de tarefas | Adicionar tarefas com responsável, prazo e status |
| ✅ Registro de decisões | Armazenar decisões importantes do grupo |
| 📄 Documento colaborativo | Criar e atualizar seções do planejamento |
| 📊 Quadro do projeto | Resumo completo do estado atual |
| 🧠 Fluxo ReAct | Agente decide dinamicamente quando usar ferramentas |

---

## 🏗️ Estrutura do Projeto

```
AcademicFlow-Agent/
├── React_Project.py        # Agente principal com grafo LangGraph
├── ui.py                   # Interface do usuário
├── requirements.txt        # Dependências de produção
├── requirements-dev.txt    # Dependências de desenvolvimento
├── tests/
│   ├── conftest.py                    # Fixtures de teste
│   ├── test_agent_configuration.py    # Testes de configuração do agente
│   └── test_collaboration_tools.py    # Testes das ferramentas
└── README.md
```

---

## 🛠️ Como Executar

**Pré-requisitos:** Python 3.8+ e [Ollama](https://ollama.ai) instalado.

```bash
# 1. Clone o repositório
git clone https://github.com/Pedrosasouza/AcademicFlow-Agent.git
cd AcademicFlow-Agent

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Baixe o modelo LLM
ollama pull llama3.2

# 4. Execute o agente
python React_Project.py
```

---

## 🧪 Testes

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## 📚 Aprendizados

- Construção de **agentes autônomos** com ciclos de raciocínio (ReAct)
- Orquestração de fluxos complexos com **grafos de estado** (LangGraph)
- Integração de **LLMs locais** para desenvolvimento sem custos de API
- Implementação de **Tool Calling** para extensibilidade do agente
- Escrita de **testes automatizados** para sistemas baseados em IA

---

## 📝 Licença

Este projeto é de uso acadêmico e pessoal.

---

<p align="center">
  Desenvolvido por <a href="https://github.com/Pedrosasouza"><strong>Pedro Santos Souza</strong></a> · USP São Carlos
</p>
