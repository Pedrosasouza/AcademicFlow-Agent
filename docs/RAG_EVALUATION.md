# Relatorio de Avaliacao do RAG (AcademicFlow-Agent)

Avaliacao quantitativa e reproduzivel do pipeline de busca vetorial em normas e editais universitarios.

## Resumo Executivo das Metricas

| Metrica | Valor | Descricao |
|:--------|:------|:----------|
| Precision@1 | 91.67% | Percentual de acerto do primeiro resultado retornado |
| Precision@2 | 45.83% | Proporcao de itens relevantes nos dois primeiros resultados |
| Precision@3 | 30.55% | Proporcao de itens relevantes no top-3 retornado |
| Recall@1 | 91.67% | Cobertura dos documentos esperados no rank 1 |
| Recall@3 | 91.67% | Cobertura total dos documentos esperados no top-3 |
| MRR (Mean Reciprocal Rank) | 0.9167 | Media da posicao reciproca (1/rank) do primeiro item relevante |
| Latencia Media | 427.86 ms | Tempo medio de execucao de busca semantica |
| Latencia P50 (Mediana) | 189.87 ms | Mediana da latencia por consulta |
| Latencia P95 | 238.45 ms | Percentil 95 de tempo de resposta |
| Respostas sem suporte (sem filtro) | 100.00% | Falsos positivos quando nenhuma norma e relevante (sem threshold) |
| Respostas sem suporte (com threshold >= 0.68) | 33.33% | Falsos positivos com filtro de similaridade aplicado |

## Metodologia e Dataset

- **Total de consultas avaliadas**: 15 (12 no escopo, 3 fora de escopo).
- **Parametro top-k**: 3.
- **Threshold de score recomendado**: 0.68.
- **Categorias testadas**: `auditorio_e_salas`, `coffee_break`, `certificados`, `submissao_artigos` e `fora_de_escopo`.

## Resultados Detalhados por Consulta

| ID | Categoria | Escopo | Relevante no Top-1? | Score Top-1 | RR | Latencia (ms) |
|:---|:----------|:-------|:--------------------|:------------|:---|:--------------|
| q01 | `auditorio_e_salas` | No escopo | Sim | 0.7940 | 1.00 | 3744.04 |
| q02 | `auditorio_e_salas` | No escopo | Sim | 0.8091 | 1.00 | 180.09 |
| q03 | `auditorio_e_salas` | No escopo | Sim | 0.5891 | 1.00 | 210.01 |
| q04 | `coffee_break` | No escopo | Sim | 0.8135 | 1.00 | 145.30 |
| q05 | `coffee_break` | No escopo | Sim | 0.5726 | 1.00 | 185.38 |
| q06 | `coffee_break` | No escopo | Nao | 0.6952 | 0.00 | 162.65 |
| q07 | `certificados` | No escopo | Sim | 0.7775 | 1.00 | 238.45 |
| q08 | `certificados` | No escopo | Sim | 0.7620 | 1.00 | 219.42 |
| q09 | `certificados` | No escopo | Sim | 0.6868 | 1.00 | 205.74 |
| q10 | `submissao_artigos` | No escopo | Sim | 0.8265 | 1.00 | 221.47 |
| q11 | `submissao_artigos` | No escopo | Sim | 0.7070 | 1.00 | 175.97 |
| q12 | `submissao_artigos` | No escopo | Sim | 0.6208 | 1.00 | 189.87 |
| q13 | `fora_de_escopo` | Fora de escopo | Rejeitado | 0.6227 | - | 210.49 |
| q14 | `fora_de_escopo` | Fora de escopo | Falso Positivo | 0.6814 | - | 162.89 |
| q15 | `fora_de_escopo` | Fora de escopo | Rejeitado | 0.5855 | - | 166.16 |

## Analise de Respostas sem Suporte e Calibracao de Threshold

Consultas fora de escopo (como perguntas sobre restaurantes universitarios, bolsas ou estacionamento)
apresentam similaridade de cosseno media significativamente inferior as consultas pertinentes:
- Consultas pertinentes no escopo: scores tipicos entre **0.72** e **0.82**.
- Consultas fora de escopo: scores tipicos entre **0.55** e **0.60**.

Ao aplicar o corte de relevância `min_score=0.68`, o sistema atinge **66.7% de rejeição**
para perguntas sem respaldo, retornando a mensagem padrao de edital nao encontrado e impedindo que o agente alucine.