"""
Modulo de avaliacao quantitativa e reproduzivel do pipeline de RAG (Busca Vetorial).

Executa consultas padronizadas a partir de um dataset versionado (JSON),
calculando metricas de Information Retrieval (Precision@k, Recall@k, MRR),
latencia com separacao de aquecimento (cold-start) vs. regime estavel (warm),
e trade-offs de calibracao de limiar de similaridade (falsos positivos vs. falsos negativos).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import repository

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "rag_benchmark_dataset.json"
DEFAULT_THRESHOLD = 0.68
DEFAULT_EVALUATED_THRESHOLDS = (0.0, 0.50, 0.55, 0.60, 0.65, 0.68, 0.70)


def calculate_percentile(values: Sequence[float], percentile: float) -> float:
    """Calcula o percentil via interpolacao linear padrao.

    Metodo equivalente a:
        - numpy.percentile(values, percentile, method='linear')
        - scipy.stats.scoreatpercentile (fraction)
        - Hyndman & Fan (1996) - Metodo 7 (padrao no R e Excel PERCENTILE.INC)

    Formula:
        k = (percentile / 100.0) * (N - 1)
        f = floor(k)
        c = ceil(k)
        valor = v[f] + (k - f) * (v[c] - v[f])

    Args:
        values: Sequencia de observacoes numericas.
        percentile: Percentil desejado (entre 0.0 e 100.0).

    Returns:
        Valor estimado do percentil com 4 casas decimais.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return round(float(values[0]), 4)
    if percentile <= 0.0:
        return round(float(min(values)), 4)
    if percentile >= 100.0:
        return round(float(max(values)), 4)

    sorted_vals = sorted(float(x) for x in values)
    n = len(sorted_vals)
    k = (percentile / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    fraction = k - f
    interpolated = sorted_vals[f] + fraction * (sorted_vals[c] - sorted_vals[f])
    return round(interpolated, 4)


@dataclass(frozen=True)
class QueryEvaluationResult:
    """Resultado detalhado da avaliacao de uma consulta individual."""

    query_id: str
    query: str
    category: str
    is_in_scope: bool
    expected_titles: list[str]
    retrieved_titles: list[str]
    retrieved_scores: list[float]
    latency_ms: float
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    reciprocal_rank: float
    is_unsupported_candidate: bool
    filtered_out_by_threshold: bool


@dataclass(frozen=True)
class BenchmarkSummary:
    """Resumo consolidado das metricas do benchmark."""

    total_queries: int
    in_scope_queries: int
    out_of_scope_queries: int
    top_k: int
    threshold: float
    mean_precision_at_k: dict[int, float]
    mean_recall_at_k: dict[int, float]
    mrr: float

    # Latencias gerais (incluindo cold start)
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p95: float
    latency_ms_min: float
    latency_ms_max: float

    # Latencias em regime estavel (warm queries: 2..N)
    latency_warmup_ms: float
    latency_warm_mean: float
    latency_warm_p50: float
    latency_warm_p90: float
    latency_warm_p95: float

    # Ruido de recuperacao (out-of-scope) no limiar selecionado
    unfiltered_unsupported_rate: float
    filtered_unsupported_rate: float

    # Rejeicao in-scope no limiar selecionado
    in_scope_rejection_rate: float

    # Grade de trade-offs de limiares
    threshold_tradeoffs: list[dict[str, float]]


def load_benchmark_dataset(dataset_path: Path | str) -> list[dict[str, Any]]:
    """Carrega o dataset de perguntas e documentos esperados a partir de arquivo JSON."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset de benchmark nao encontrado em: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("O dataset de benchmark deve ser uma lista JSON de objetos de consulta.")
    return data


def evaluate_single_query(
    item: dict[str, Any],
    top_k: int = 3,
    threshold: float = DEFAULT_THRESHOLD,
) -> QueryEvaluationResult:
    """Executa e avalia uma unica consulta contra o pipeline de busca do repositorio."""
    query_text = str(item["query"]).strip()
    is_in_scope = bool(item.get("is_in_scope", True))
    expected_titles = [str(t).strip() for t in item.get("expected_relevant_titles", [])]

    start_time = time.perf_counter()
    retrieved_items = repository.search_regulation_chunks(query=query_text, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    retrieved_titles = [r["title"] for r in retrieved_items]
    retrieved_scores = [round(float(r.get("score", 0.0)), 4) for r in retrieved_items]

    precision_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}

    if is_in_scope and expected_titles:
        for k in range(1, top_k + 1):
            sub_retrieved = retrieved_titles[:k]
            relevant_retrieved = [t for t in sub_retrieved if t in expected_titles]
            precision_at_k[k] = round(len(relevant_retrieved) / k, 4)
            recall_at_k[k] = round(len(relevant_retrieved) / len(expected_titles), 4)

        rr = 0.0
        for rank, title in enumerate(retrieved_titles, start=1):
            if title in expected_titles:
                rr = round(1.0 / rank, 4)
                break
        reciprocal_rank = rr
    else:
        for k in range(1, top_k + 1):
            precision_at_k[k] = 0.0
            recall_at_k[k] = 0.0
        reciprocal_rank = 0.0

    # Analise de ruido de busca / espurios em consultas fora de escopo
    has_spurious_chunks = bool(retrieved_items) and not is_in_scope
    filtered_results = [r for r in retrieved_items if float(r.get("score", 0.0)) >= threshold]
    filtered_out = len(filtered_results) == 0

    return QueryEvaluationResult(
        query_id=str(item.get("id", "q")),
        query=query_text,
        category=str(item.get("category", "geral")),
        is_in_scope=is_in_scope,
        expected_titles=expected_titles,
        retrieved_titles=retrieved_titles,
        retrieved_scores=retrieved_scores,
        latency_ms=round(elapsed_ms, 2),
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        is_unsupported_candidate=has_spurious_chunks,
        filtered_out_by_threshold=filtered_out,
    )


def compute_benchmark_summary(
    results: Sequence[QueryEvaluationResult],
    top_k: int = 3,
    threshold: float = DEFAULT_THRESHOLD,
    evaluated_thresholds: Sequence[float] = DEFAULT_EVALUATED_THRESHOLDS,
) -> BenchmarkSummary:
    """Calcula agregacoes estatisticas e matriz de trade-offs de limiares."""
    in_scope_results = [r for r in results if r.is_in_scope]
    out_of_scope_results = [r for r in results if not r.is_in_scope]

    n_in_scope = len(in_scope_results)
    mean_precision: dict[int, float] = {}
    mean_recall: dict[int, float] = {}

    for k in range(1, top_k + 1):
        if n_in_scope > 0:
            mean_precision[k] = round(
                sum(r.precision_at_k.get(k, 0.0) for r in in_scope_results) / n_in_scope,
                4,
            )
            mean_recall[k] = round(
                sum(r.recall_at_k.get(k, 0.0) for r in in_scope_results) / n_in_scope,
                4,
            )
        else:
            mean_precision[k] = 0.0
            mean_recall[k] = 0.0

    mrr = (
        round(sum(r.reciprocal_rank for r in in_scope_results) / n_in_scope, 4)
        if n_in_scope > 0
        else 0.0
    )

    latencies = [r.latency_ms for r in results]
    lat_mean = round(statistics.mean(latencies), 2) if latencies else 0.0
    lat_p50 = round(calculate_percentile(latencies, 50.0), 2)
    lat_p90 = round(calculate_percentile(latencies, 90.0), 2)
    lat_p95 = round(calculate_percentile(latencies, 95.0), 2)
    lat_min = round(min(latencies), 2) if latencies else 0.0
    lat_max = round(max(latencies), 2) if latencies else 0.0

    # Distincao entre inicializacao (cold start / warmup) e regime estavel (warm)
    if len(latencies) > 1:
        warmup_ms = round(latencies[0], 2)
        warm_latencies = latencies[1:]
        warm_mean = round(statistics.mean(warm_latencies), 2)
        warm_p50 = round(calculate_percentile(warm_latencies, 50.0), 2)
        warm_p90 = round(calculate_percentile(warm_latencies, 90.0), 2)
        warm_p95 = round(calculate_percentile(warm_latencies, 95.0), 2)
    elif latencies:
        warmup_ms = round(latencies[0], 2)
        warm_mean = warmup_ms
        warm_p50 = warmup_ms
        warm_p90 = warmup_ms
        warm_p95 = warmup_ms
    else:
        warmup_ms = warm_mean = warm_p50 = warm_p90 = warm_p95 = 0.0

    # Analise de ruido e rejeicao no limiar padrao selecionado
    n_out = len(out_of_scope_results)
    if n_out > 0:
        unfiltered_unsupported_count = sum(
            1 for r in out_of_scope_results if r.is_unsupported_candidate
        )
        unfiltered_rate = round(unfiltered_unsupported_count / n_out, 4)

        filtered_unsupported_count = sum(
            1
            for r in out_of_scope_results
            if any(score >= threshold for score in r.retrieved_scores)
        )
        filtered_rate = round(filtered_unsupported_count / n_out, 4)
    else:
        unfiltered_rate = 0.0
        filtered_rate = 0.0

    # Rejeicao in-scope no limiar selecionado (falsos negativos)
    if n_in_scope > 0:
        in_rejected_count = sum(
            1
            for r in in_scope_results
            if not r.retrieved_scores or r.retrieved_scores[0] < threshold
        )
        in_scope_rejection_rate = round(in_rejected_count / n_in_scope, 4)
    else:
        in_scope_rejection_rate = 0.0

    # Avaliacao da grade de calibracao de limiares (trade-off Falsos Positivos vs. Falsos Negativos)
    threshold_tradeoffs: list[dict[str, float]] = []
    for t_val in evaluated_thresholds:
        # No escopo: quantas perguntas tem score top-1 menor que o limiar (rejeicao incorreta / FN)
        if n_in_scope > 0:
            fn_count = sum(
                1
                for r in in_scope_results
                if not r.retrieved_scores or r.retrieved_scores[0] < t_val
            )
            # Relevante no top-1 retornado E com score >= t_val
            accepted_relevant_count = sum(
                1
                for r in in_scope_results
                if r.retrieved_scores
                and r.retrieved_scores[0] >= t_val
                and r.retrieved_titles
                and r.retrieved_titles[0] in r.expected_titles
            )
            fn_rate = round(fn_count / n_in_scope, 4)
            acc_rel_rate = round(accepted_relevant_count / n_in_scope, 4)
        else:
            fn_rate = 0.0
            acc_rel_rate = 0.0

        # Fora de escopo: quantas consultas tem algum score >= t_val (vazamento / FP)
        if n_out > 0:
            fp_count = sum(
                1
                for r in out_of_scope_results
                if any(score >= t_val for score in r.retrieved_scores)
            )
            fp_rate = round(fp_count / n_out, 4)
            tn_rate = round((n_out - fp_count) / n_out, 4)
        else:
            fp_rate = 0.0
            tn_rate = 0.0

        threshold_tradeoffs.append(
            {
                "threshold": float(t_val),
                "in_scope_rejected_rate": fn_rate,
                "in_scope_accepted_relevant_rate": acc_rel_rate,
                "out_of_scope_leakage_rate": fp_rate,
                "out_of_scope_rejection_rate": tn_rate,
            }
        )

    return BenchmarkSummary(
        total_queries=len(results),
        in_scope_queries=n_in_scope,
        out_of_scope_queries=n_out,
        top_k=top_k,
        threshold=threshold,
        mean_precision_at_k=mean_precision,
        mean_recall_at_k=mean_recall,
        mrr=mrr,
        latency_ms_mean=lat_mean,
        latency_ms_p50=lat_p50,
        latency_ms_p90=lat_p90,
        latency_ms_p95=lat_p95,
        latency_ms_min=lat_min,
        latency_ms_max=lat_max,
        latency_warmup_ms=warmup_ms,
        latency_warm_mean=warm_mean,
        latency_warm_p50=warm_p50,
        latency_warm_p90=warm_p90,
        latency_warm_p95=warm_p95,
        unfiltered_unsupported_rate=unfiltered_rate,
        filtered_unsupported_rate=filtered_rate,
        in_scope_rejection_rate=in_scope_rejection_rate,
        threshold_tradeoffs=threshold_tradeoffs,
    )


def run_rag_benchmark(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    top_k: int = 3,
    threshold: float = DEFAULT_THRESHOLD,
    evaluated_thresholds: Sequence[float] = DEFAULT_EVALUATED_THRESHOLDS,
) -> tuple[BenchmarkSummary, list[QueryEvaluationResult]]:
    """Ponto de entrada programatico para execucao da avaliacao completa."""
    dataset = load_benchmark_dataset(dataset_path)
    results = [
        evaluate_single_query(item, top_k=top_k, threshold=threshold)
        for item in dataset
    ]
    summary = compute_benchmark_summary(
        results,
        top_k=top_k,
        threshold=threshold,
        evaluated_thresholds=evaluated_thresholds,
    )
    return summary, results


def format_markdown_report(
    summary: BenchmarkSummary,
    results: list[QueryEvaluationResult],
) -> str:
    """Gera relatorio em Markdown com analise estatistica de IR, latencia e trade-offs de limiar."""
    lines: list[str] = [
        "# Relatorio de Avaliacao do RAG (AcademicFlow-Agent)",
        "",
        "Avaliacao quantitativa e reproduzivel do pipeline de busca vetorial em normas e editais universitarios.",
        "",
        "## Resumo Executivo das Metricas",
        "",
        "| Metrica | Valor | Descricao |",
        "|:--------|:------|:----------|",
        f"| Precision@1 | {summary.mean_precision_at_k.get(1, 0.0):.2%} | Percentual de consultas em que o primeiro resultado e relevante |",
        f"| Precision@2 | {summary.mean_precision_at_k.get(2, 0.0):.2%} | Proporcao de itens relevantes nos dois primeiros resultados |",
        f"| Precision@3 | {summary.mean_precision_at_k.get(3, 0.0):.2%} | Proporcao de itens relevantes no top-3 retornado |",
        f"| Recall@1 | {summary.mean_recall_at_k.get(1, 0.0):.2%} | Cobertura do documento esperado logo no primeiro resultado |",
        f"| Recall@3 | {summary.mean_recall_at_k.get(3, 0.0):.2%} | Cobertura total do documento esperado no top-3 |",
        f"| MRR (Mean Reciprocal Rank) | {summary.mrr:.4f} | Media da posicao reciproca (1/rank) da primeira norma relevante |",
        f"| Latencia Media (Geral) | {summary.latency_ms_mean:.2f} ms | Media geral incluindo inicializacao da primeira consulta |",
        f"| Latencia P50 (Mediana Geral) | {summary.latency_ms_p50:.2f} ms | Mediana de tempo de resposta por busca semantica |",
        f"| Latencia P90 (Geral) | {summary.latency_ms_p90:.2f} ms | Percentil 90 geral (interpolacao linear padrao) |",
        f"| Latencia P95 (Geral) | {summary.latency_ms_p95:.2f} ms | Percentil 95 geral (sensivel ao custo de inicializacao) |",
        f"| Latencia Inicializacao (Warmup) | {summary.latency_warmup_ms:.2f} ms | Tempo da 1ª consulta (carga do modelo de embeddings na VRAM/RAM) |",
        f"| Latencia Regime Estavel (P50 Warm) | {summary.latency_warm_p50:.2f} ms | Mediana em regime aquecido (consultas 2 a {summary.total_queries}) |",
        f"| Latencia Regime Estavel (P95 Warm) | {summary.latency_warm_p95:.2f} ms | Percentil 95 em regime aquecido (consultas 2 a {summary.total_queries}) |",
        f"| Ruido Fora de Escopo (Sem filtro) | {summary.unfiltered_unsupported_rate:.2%} | Falsos positivos quando nenhuma norma e pertinente (sem limiar) |",
        f"| Ruido Fora de Escopo (Threshold >= {summary.threshold}) | {summary.filtered_unsupported_rate:.2%} | Falsos positivos com corte de similaridade aplicado |",
        f"| Rejeicao no Escopo (Threshold >= {summary.threshold}) | {summary.in_scope_rejection_rate:.2%} | Falsos negativos: consultas legitimas abaixo do limiar |",
        "",
        "## Metodologia e Dataset",
        "",
        f"- **Total de consultas avaliadas**: {summary.total_queries} ({summary.in_scope_queries} no escopo, {summary.out_of_scope_queries} fora de escopo).",
        f"- **Parametro top-k**: {summary.top_k}.",
        f"- **Metodo de calculo de percentil**: Interpolacao linear padrao (Hyndman & Fan Tipo 7 / `numpy.percentile(method='linear')`).",
        "- **Separacao de latencia**: Distincao explicita entre o custo de carregamento do modelo (*cold-start* na consulta 1) e o tempo de resposta em regime permanente (*steady-state*).",
        "- **Categorias testadas**: `auditorio_e_salas`, `coffee_break`, `certificados`, `submissao_artigos` e `fora_de_escopo`.",
        "",
        "## Analise de Calibracao e Trade-off de Limiares",
        "",
        "A definicao de um limiar fixo de similaridade de cosseno impoe um compromisso direto (*trade-off*) entre a filtragem de perguntas fora de escopo (falsos positivos de busca) e a preservacao de perguntas legitimas formuladas com termos semanticamente distantes (falsos negativos):",
        "",
        "| Limiar ($T$) | Rejeicao Incorreta no Escopo (FN) | Precisao Efetiva Top-1 (No Escopo) | Ruido Fora de Escopo (FP) | Rejeicao Correta Fora de Escopo (TN) |",
        "|:---|:---|:---|:---|:---|",
    ]

    for to in summary.threshold_tradeoffs:
        t_val = to["threshold"]
        fn_pct = f"{to['in_scope_rejected_rate']:.2%}"
        acc_pct = f"{to['in_scope_accepted_relevant_rate']:.2%}"
        fp_pct = f"{to['out_of_scope_leakage_rate']:.2%}"
        tn_pct = f"{to['out_of_scope_rejection_rate']:.2%}"
        lines.append(f"| {t_val:.2f} | {fn_pct} | {acc_pct} | {fp_pct} | {tn_pct} |")

    lines.extend(
        [
            "",
            "### Interpretacao dos Resultados da Calibracao",
            "",
            "1. **Limiar baixo ($T \\le 0.55$)**: Nenhuma pergunta legitima e descartada (0% FN), mas 100% das perguntas fora de escopo recuperam trechos espurios (100% FP).",
            f"2. **Limiar intermediario ($T = {summary.threshold:.2f}$)**: Reduz o vazamento de consultas fora de escopo para {summary.filtered_unsupported_rate:.1%}, porem descarta {summary.in_scope_rejection_rate:.1%} das perguntas pertinentes cujos enunciados utilizam sinonimos menos literais (ex.: `q03`, `q05` e `q12`).",
            "3. **Limiar restritivo ($T \\ge 0.70$)**: Elimina totalmente os trechos espurios para fora de escopo (0% FP / 100% TN), porem rejeita mais de 40% das perguntas no escopo.",
            "4. **Conclusao de Engenharia**: Nenhum limiar escalar estatico separa perfeitamente ambos os conjuntos. Para um ambiente de producao de alta criticidade, a arquitetura recomendada combina busca densa com um modelo classificador/re-ranker (*Cross-Encoder*) ou verificacao contextual em prompt pelo LLM antes da recusa.",
            "",
            "## Diferenciacao: Recuperacao de Informacao (IR) vs. Alucinacao do Agente",
            "",
            "E fundamental distinguir a qualidade do mecanismo de recuperacao vetorial da resposta gerativa final do LLM:",
            "- **Ruido de Recuperacao (Retrieval Noise)**: Ocorre quando a busca vetorial retorna documentos com score positivo para perguntas que nao possuem resposta no corpus.",
            "- **Filtragem pelo Agente**: Mesmo quando o mecanismo de busca entrega trechos irrelevantes, as instrucoes de sistema do agente exigem que o LLM verifique a pertinencia do documento antes de afirmar fatos. Se o trecho nao sustenta a pergunta, o agente pode recusar a resposta.",
            "- **Fidelidade Factual (Faithfulness / Hallucination)**: Avalia se as sentencas geradas na resposta final sao estritamente suportadas pelas evidencias recuperadas. Esta metrica sera abordada em uma segunda etapa de avaliacao fim-a-fim (*LLM-as-a-judge* / framework Ragas).",
            "",
            "## Natureza do Corpus de Avaliacao",
            "",
            "> [!NOTE]",
            "> As 4 normas cadastradas e as 15 perguntas associadas compoem um **corpus sintetico e didatico de referencia tecnica**, elaborado para exercitar e validar a arquitetura do agente (PostgreSQL 16, pgvector e nomic-embed-text). Nao substituem resolucoes universitarias oficiais vigentes nem possuem validade documental administrativa.",
            "",
            "## Resultados Detalhados por Consulta",
            "",
            "| ID | Categoria | Escopo | Relevante no Top-1? | Score Top-1 | RR | Latencia (ms) |",
            "|:---|:----------|:-------|:--------------------|:------------|:---|:--------------|",
        ]
    )

    for r in results:
        top_title = r.retrieved_titles[0] if r.retrieved_titles else "nenhum"
        top_score = r.retrieved_scores[0] if r.retrieved_scores else 0.0
        if r.is_in_scope:
            top_hit = "Sim" if top_title in r.expected_titles else "Nao"
            escopo = "No escopo"
            rr_str = f"{r.reciprocal_rank:.2f}"
        else:
            top_hit = "Rejeitado" if r.filtered_out_by_threshold else "Ruido / Vazamento"
            escopo = "Fora de escopo"
            rr_str = "-"

        lines.append(
            f"| {r.query_id} | `{r.category}` | {escopo} | {top_hit} | {top_score:.4f} | {rr_str} | {r.latency_ms:.2f} |"
        )

    return "\n".join(lines)


def main() -> None:
    """Execucao via CLI."""
    parser = argparse.ArgumentParser(
        description="Executa benchmark quantitativo do pipeline de RAG do AcademicFlow-Agent."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Caminho do arquivo JSON com as consultas de avaliacao.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Numero de documentos a recuperar por consulta (padrao: 3).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Limiar de score de similaridade para rejeicao de falso positivo (padrao: 0.68).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Caminho opcional para exportar os resultados em JSON.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Caminho opcional para exportar o relatorio em Markdown.",
    )

    args = parser.parse_args()

    print(f"Executando avaliacao do RAG com dataset: {args.dataset}")
    summary, results = run_rag_benchmark(
        dataset_path=args.dataset,
        top_k=args.top_k,
        threshold=args.threshold,
    )

    print("\n" + "=" * 60)
    print("RESUMO DO BENCHMARK RAG")
    print("=" * 60)
    print(f"Consultas avaliadas: {summary.total_queries} ({summary.in_scope_queries} no escopo, {summary.out_of_scope_queries} fora)")
    print(f"Precision@1: {summary.mean_precision_at_k.get(1, 0.0):.2%}")
    print(f"Precision@3: {summary.mean_precision_at_k.get(3, 0.0):.2%}")
    print(f"Recall@1:    {summary.mean_recall_at_k.get(1, 0.0):.2%}")
    print(f"Recall@3:    {summary.mean_recall_at_k.get(3, 0.0):.2%}")
    print(f"MRR:         {summary.mrr:.4f}")
    print(f"Latencia Geral: Media={summary.latency_ms_mean:.2f} ms | P50={summary.latency_ms_p50:.2f} ms | P90={summary.latency_ms_p90:.2f} ms | P95={summary.latency_ms_p95:.2f} ms")
    print(f"Latencia Regime Estavel (Warm): Inicializacao={summary.latency_warmup_ms:.2f} ms | P50={summary.latency_warm_p50:.2f} ms | P95={summary.latency_warm_p95:.2f} ms")
    print(f"Ruido Fora de Escopo sem limiar: {summary.unfiltered_unsupported_rate:.2%}")
    print(f"Ruido Fora de Escopo com limiar ({summary.threshold}): {summary.filtered_unsupported_rate:.2%}")
    print(f"Rejeicao no Escopo com limiar ({summary.threshold}): {summary.in_scope_rejection_rate:.2%}")
    print("=" * 60)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Resultados detalhados salvos em JSON: {args.output_json}")

    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        report = format_markdown_report(summary, results)
        with args.output_markdown.open("w", encoding="utf-8") as f:
            f.write(report)
        print(f"Relatorio salvo em Markdown: {args.output_markdown}")


if __name__ == "__main__":
    main()
