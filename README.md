# AcademicFlow-Agent

Sistema colaborativo baseado em agentes de IA utilizando arquitetura ReAct com LangGraph para auxiliar no planejamento de eventos acadêmicos.

## Sobre o Projeto

O AcademicFlow-Agent é um protótipo de agente colaborativo capaz de auxiliar grupos durante o planejamento de eventos acadêmicos. O sistema utiliza um fluxo ReAct (Reason + Act), permitindo que o modelo de linguagem decida entre responder diretamente ao usuário ou executar ferramentas específicas para organizar informações do projeto.

O agente pode:

- Registrar tarefas;
- Registrar decisões importantes;
- Atualizar seções de documentos colaborativos;
- Consultar o estado atual do planejamento;
- Auxiliar participantes durante discussões e organização do evento.

O foco principal do projeto é demonstrar a construção de agentes inteligentes utilizando LangGraph, integração com LLMs locais e uso de ferramentas em tempo de execução.

---

## Arquitetura

O sistema foi desenvolvido utilizando:

- ReAct Agent Pattern
- LangGraph State Graph
- Tool Calling
- LLM local com Ollama
- Memória simples em estrutura Python

Fluxo principal:

1. Usuário envia uma mensagem;
2. O agente interpreta a intenção;
3. Decide entre:
   - responder diretamente;
   - chamar ferramentas;
4. O estado do projeto é atualizado;
5. O agente retorna uma resposta contextualizada.

---

## Funcionalidades

### Registro de tarefas
Permite adicionar tarefas com responsável, prazo e status.

### Registro de decisões
Armazena decisões importantes tomadas pelo grupo.

### Documento colaborativo
Criação e atualização de seções do planejamento do evento.

### Consulta do quadro do projeto
Resumo completo do estado atual do planejamento.

### Fluxo ReAct
O agente decide dinamicamente quando utilizar ferramentas.

---

## Tecnologias Utilizadas

- Python
- LangGraph
- LangChain
- Ollama
- Llama 3.2
- HTTPX

---

## Estrutura do Projeto

```bash
academicflow-agent/
│
├── React_Project.py
├── README.md

```

---

## Como Executar

### Pré-requisitos

Certifique-se de ter instalado:
- Python 3.8+
- Ollama (download em [ollama.ai](https://ollama.ai))

### Passo 1: Clonar ou acessar o projeto

```bash
cd AcademicFlow-Agent
```

### Passo 2: Criar um ambiente virtual (opcional, mas recomendado)

```bash
python -m venv venv
```

**No Windows:**
```bash
venv\Scripts\activate
```

**No Linux/Mac:**
```bash
source venv/bin/activate
```

### Passo 3: Instalar as dependências

```bash
pip install langgraph langchain-ollama langchain-core httpx streamlit
```

### Passo 4: Baixar e configurar o modelo Ollama

Primeiro, instale o Ollama (se ainda não tiver) em [ollama.ai](https://ollama.ai).

Depois, em um terminal separado, puxe o modelo:

```bash
ollama pull llama3.2
```

### Passo 5: Iniciar o Ollama

Em um terminal separado (manter aberto durante a execução do projeto):

```bash
ollama serve
```

O Ollama será iniciado em `http://localhost:11434`

### Passo 6: Executar a aplicação

Com o Ollama rodando em outro terminal, execute:

```bash
streamlit run ui.py
```

A aplicação será aberta automaticamente no navegador em `http://localhost:8501`

---

## Uso

1. **Envie mensagens** para o agente utilizando o chat da interface Streamlit
2. **O agente pode:**
   - Registrar tarefas para o evento acadêmico
   - Registrar decisões importantes
   - Atualizar o documento colaborativo
   - Consultar o estado atual do planejamento

3. **Monitore as atividades** no painel lateral (sidebar) que mostra:
   - Ferramentas chamadas pelo agente
   - Resumo do quadro do projeto (tarefas, decisões, seções do documento)

---

## Estrutura dos Arquivos

- **React_Projetct.py**: Contém o agente ReAct com LangGraph, definição das ferramentas e lógica do fluxo
- **ui.py**: Interface Streamlit para interagir com o agente
- **README.md**: Este arquivo

---

## Troubleshooting

### Erro: "Connection refused" ao conectar com Ollama
- Certifique-se de que o Ollama está rodando com `ollama serve` em outro terminal
- Verifique se está acessível em `http://localhost:11434`

### Erro: Modelo não encontrado
- Execute `ollama pull llama3.2` para baixar o modelo

### Interface Streamlit não abre
- Verifique se está na pasta do projeto
- Execute `streamlit run ui.py` com as dependências instaladas corretamente

## Exemplo de Uso

```text
Grupo: Criar tarefa para reservar o auditório até sexta.
Agente: Tarefa registrada com sucesso.
```

```text
Grupo: Qual o estado atual do projeto?
Agente: Exibe tarefas, decisões e seções registradas.
```

---

## Conceitos Demonstrados

- Arquitetura de Agentes de IA
- ReAct Pattern
- Tool Calling
- State Management
- Planejamento colaborativo assistido por IA
- Integração de LLM local
- Fluxos multi-step com LangGraph

---

## Possíveis Melhorias Futuras

- Persistência em banco de dados;
- Múltiplos agentes especializados;
- Integração com calendário;
- Sistema de autenticação;
- Memória vetorial;
- Upload de arquivos e atas.
