# Templates de Teste

Este arquivo serve como roteiro de testes manuais e de aceitacao para o
AcademicFlow-Agent. Os testes unitarios automatizados ficam na pasta `tests/`.

## Template 1 - Comunicacao entre usuarios e agente

**Objetivo:** verificar se a interface permite enviar mensagens identificadas por usuario.

**Pre-condicoes:**
- Streamlit executando com `streamlit run ui.py`.
- Ollama executando com `ollama serve`.
- Modelo `llama3.2` instalado.

**Passos:**
1. Abrir `http://localhost:8501`.
2. Selecionar o usuario `Alice` na sidebar.
3. Enviar a mensagem: `Precisamos reservar o auditorio para sexta.`
4. Selecionar o usuario `Bob`.
5. Enviar a mensagem: `Tambem precisamos convidar os palestrantes.`

**Resultado esperado:**
- As mensagens aparecem no chat.
- Cada mensagem enviada ao agente contem identificacao do usuario no historico.
- O agente responde sem perder o contexto da conversa.

## Template 2 - Coordenacao por registro de tarefas

**Objetivo:** verificar se o agente consegue registrar tarefas do planejamento.

**Passos:**
1. Enviar: `Crie uma tarefa para reservar o auditorio. Responsavel Alice. Prazo sexta-feira.`
2. Observar a sidebar em `Quadro do projeto`.

**Resultado esperado:**
- A ferramenta `register_task` e chamada.
- O contador de tarefas aumenta.
- A tarefa aparece com titulo, responsavel e prazo.

## Template 3 - Colaboracao por registro de decisoes

**Objetivo:** verificar se decisoes coletivas podem ser registradas.

**Passos:**
1. Enviar: `Registre a decisao: o evento sera no auditorio principal. Motivo: comporta mais participantes.`
2. Solicitar: `Qual o estado atual do projeto?`

**Resultado esperado:**
- A ferramenta `register_decision` e chamada.
- A decisao aparece no resumo do projeto.
- O agente usa a decisao registrada ao responder sobre o estado atual.

## Template 4 - Documento colaborativo

**Objetivo:** verificar se o agente consegue criar ou atualizar secoes do documento do evento.

**Passos:**
1. Enviar: `Crie uma secao chamada Objetivos com o texto: Organizar uma semana academica com palestras e oficinas.`
2. Enviar: `Consulte o quadro do projeto.`

**Resultado esperado:**
- A ferramenta `update_document_section` e chamada.
- O quadro mostra uma secao chamada `Objetivos`.
- A consulta informa que existe uma secao registrada.

## Template 5 - Recuperacao apos erro de Ollama

**Objetivo:** verificar se a interface mostra erro compreensivel quando o Ollama esta fechado.

**Passos:**
1. Fechar o processo do Ollama.
2. Executar a interface Streamlit.
3. Enviar qualquer mensagem no chat.

**Resultado esperado:**
- A interface exibe uma mensagem de erro orientando a executar `ollama serve`.
- A aplicacao nao quebra com traceback visivel para o usuario final.

## Como Rodar os Testes Unitarios

Instale as dependencias principais e o pytest:

```powershell
pip install langgraph langchain-ollama langchain-core httpx streamlit pytest
```

Execute:

```powershell
pytest
```

Os testes unitarios nao chamam o Ollama e nao executam a interface Streamlit.
