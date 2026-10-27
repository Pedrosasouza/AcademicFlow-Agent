# Relatorio de Avaliacao do RAG (AcademicFlow-Agent)

Avaliacao quantitativa e reproduzivel do pipeline de busca vetorial em normas e editais universitarios.

## Resumo Executivo das Metricas

| Metrica | Valor | Descricao |
|:--------|:------|:----------|
| Precision@1 | 91.67% | Percentual de consultas em que o primeiro resultado e relevante |
| Precision@2 | 45.83% | Proporcao de itens relevantes nos dois primeiros resultados |
| Precision@3 | 30.55% | Proporcao de itens relevantes no top-3 retornado |
| Recall@1 | 91.67% | Cobertura do documento esperado logo no primeiro resultado |
| Recall@3 | 91.67% | Cobertura total do documento esperado no top-3 |
| MRR (Mean Reciprocal Rank) | 0.9167 | Media da posicao reciproca (1/rank) da primeira norma relevante |
| Latencia Media (Geral) | 420.03 ms | Media geral incluindo inicializacao da primeira consulta |
| Latencia P50 (Mediana Geral) | 192.50 ms | Mediana de tempo de resposta por busca semantica |
| Latencia P90 (Geral) | 248.84 ms | Percentil 90 geral (interpolacao linear padrao) |
| Latencia P95 (Geral) | 1243.28 ms | Percentil 95 geral (sensivel ao custo de inicializacao) |
| Latencia Inicializacao (Warmup) | 3563.57 ms | Tempo da 1ª consulta (carga do modelo de embeddings na VRAM/RAM) |
| Latencia Regime Estavel (P50 Warm) | 191.19 ms | Mediana em regime aquecido (consultas 2 a 15) |
| Latencia Regime Estavel (P95 Warm) | 248.82 ms | Percentil 95 em regime aquecido (consultas 2 a 15) |
| Ruido Fora de Escopo (Sem filtro) | 100.00% | Falsos positivos quando nenhuma norma e pertinente (sem limiar) |
| Ruido Fora de Escopo (Threshold >= 0.68) | 33.33% | Falsos positivos com corte de similaridade aplicado |
| Rejeicao no Escopo (Threshold >= 0.68) | 25.00% | Falsos negativos: consultas legitimas abaixo do limiar |

## Metodologia e Dataset

- **Total de consultas avaliadas**: 15 (12 no escopo, 3 fora de escopo).
- **Parametro top-k**: 3.
- **Metodo de calculo de percentil**: Interpolacao linear padrao (Hyndman & Fan Tipo 7 / `numpy.percentile(method='linear')`).
- **Separacao de latencia**: Distincao explicita entre o custo de carregamento do modelo (*cold-start* na consulta 1) e o tempo de resposta em regime permanente (*steady-state*).
- **Categorias testadas**: `auditorio_e_salas`, `coffee_break`, `certificados`, `submissao_artigos` e `fora_de_escopo`.

## Analise de Calibracao e Trade-off de Limiares

A definicao de um limiar fixo de similaridade de cosseno impoe um compromisso direto (*trade-off*) entre a filtragem de perguntas fora de escopo (falsos positivos de busca) e a preservacao de perguntas legitimas formuladas com termos semanticamente distantes (falsos negativos):

| Limiar ($T$) | Rejeicao Incorreta no Escopo (FN) | Precisao Efetiva Top-1 (No Escopo) | Ruido Fora de Escopo (FP) | Rejeicao Correta Fora de Escopo (TN) |
|:---|:---|:---|:---|:---|
| 0.00 | 0.00% | 91.67% | 100.00% | 0.00% |
| 0.50 | 0.00% | 91.67% | 100.00% | 0.00% |
| 0.55 | 0.00% | 91.67% | 100.00% | 0.00% |
| 0.60 | 16.67% | 75.00% | 66.67% | 33.33% |
| 0.65 | 25.00% | 66.67% | 33.33% | 66.67% |
| 0.68 | 25.00% | 66.67% | 33.33% | 66.67% |
| 0.70 | 41.67% | 58.33% | 0.00% | 100.00% |

### Interpretacao dos Resultados da Calibracao

1. **Limiar baixo ($T \le 0.55$)**: Nenhuma pergunta legitima e descartada (0% FN), mas 100% das perguntas fora de escopo recuperam trechos espurios (100% FP).
2. **Limiar intermediario ($T = 0.68$)**: Reduz o vazamento de consultas fora de escopo para 33.3%, porem descarta 25.0% das perguntas pertinentes cujos enunciados utilizam sinonimos menos literais (ex.: `q03`, `q05` e `q12`).
3. **Limiar restritivo ($T \ge 0.70$)**: Elimina totalmente os trechos espurios para fora de escopo (0% FP / 100% TN), porem rejeita mais de 40% das perguntas no escopo.
4. **Conclusao de Engenharia**: Nenhum limiar escalar estatico separa perfeitamente ambos os conjuntos. Para um ambiente de producao de alta criticidade, a arquitetura recomendada combina busca densa com um modelo classificador/re-ranker (*Cross-Encoder*) ou verificacao contextual em prompt pelo LLM antes da recusa.

## Diferenciacao: Recuperacao de Informacao (IR) vs. Alucinacao do Agente

E fundamental distinguir a qualidade do mecanismo de recuperacao vetorial da resposta gerativa final do LLM:
- **Ruido de Recuperacao (Retrieval Noise)**: Ocorre quando a busca vetorial retorna documentos com score positivo para perguntas que nao possuem resposta no corpus.
- **Filtragem pelo Agente**: Mesmo quando o mecanismo de busca entrega trechos irrelevantes, as instrucoes de sistema do agente exigem que o LLM verifique a pertinencia do documento antes de afirmar fatos. Se o trecho nao sustenta a pergunta, o agente pode recusar a resposta.
- **Fidelidade Factual (Faithfulness / Hallucination)**: Avalia se as sentencas geradas na resposta final sao estritamente suportadas pelas evidencias recuperadas. Esta metrica sera abordada em uma segunda etapa de avaliacao fim-a-fim (*LLM-as-a-judge* / framework Ragas).

## Natureza do Corpus de Avaliacao

> [!NOTE]
> As 4 normas cadastradas e as 15 perguntas associadas compoem um **corpus sintetico e didatico de referencia tecnica**, elaborado para exercitar e validar a arquitetura do agente (PostgreSQL 16, pgvector e nomic-embed-text). Nao substituem resolucoes universitarias oficiais vigentes nem possuem validade documental administrativa.

## Resultados Detalhados por Consulta

| ID | Categoria | Escopo | Relevante no Top-1? | Score Top-1 | RR | Latencia (ms) |
|:---|:----------|:-------|:--------------------|:------------|:---|:--------------|
| q01 | `auditorio_e_salas` | No escopo | Sim | 0.7940 | 1.00 | 3563.57 |
| q02 | `auditorio_e_salas` | No escopo | Sim | 0.8091 | 1.00 | 163.64 |
| q03 | `auditorio_e_salas` | No escopo | Sim | 0.5891 | 1.00 | 248.87 |
| q04 | `coffee_break` | No escopo | Sim | 0.8135 | 1.00 | 185.71 |
| q05 | `coffee_break` | No escopo | Sim | 0.5726 | 1.00 | 189.89 |
| q06 | `coffee_break` | No escopo | Nao | 0.6952 | 0.00 | 166.83 |
| q07 | `certificados` | No escopo | Sim | 0.7775 | 1.00 | 187.89 |
| q08 | `certificados` | No escopo | Sim | 0.7620 | 1.00 | 202.07 |
| q09 | `certificados` | No escopo | Sim | 0.6868 | 1.00 | 206.99 |
| q10 | `submissao_artigos` | No escopo | Sim | 0.8265 | 1.00 | 192.51 |
| q11 | `submissao_artigos` | No escopo | Sim | 0.7070 | 1.00 | 170.78 |
| q12 | `submissao_artigos` | No escopo | Sim | 0.6208 | 1.00 | 181.72 |
| q13 | `fora_de_escopo` | Fora de escopo | Rejeitado | 0.6227 | - | 192.50 |
| q14 | `fora_de_escopo` | Fora de escopo | Ruido / Vazamento | 0.6814 | - | 198.65 |
| q15 | `fora_de_escopo` | Fora de escopo | Rejeitado | 0.5855 | - | 248.80 |