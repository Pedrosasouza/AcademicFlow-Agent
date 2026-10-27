"""
Testes automatizados para a suíte de avaliação quantitativa do RAG (benchmarks/evaluate_rag.py).
Valida métricas (Precision@k, Recall@k, MRR), detecção de respostas sem suporte e relatório Markdown.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmarks import evaluate_rag
from benchmarks.evaluate_rag import (
    BenchmarkSummary,
    QueryEvaluationResult,
    compute_benchmark_summary,
    evaluate_single_query,
    format_markdown_report,
    load_benchmark_dataset,
    run_rag_benchmark,
)
from src import repository


def test_load_benchmark_dataset_loads_valid_json(tmp_path: Path) -> None:
    """Garante que o carregamento do dataset valida a existência e estrutura da lista JSON."""
    valid_file = tmp_path / "valid.json"
    valid_file.write_text(
        json.dumps([{"id": "q1", "query": "teste", "expected_relevant_titles": ["T1"]}]),
        encoding="utf-8",
    )

    data = load_benchmark_dataset(valid_file)
    assert len(data) == 1
    assert data[0]["id"] == "q1"

    with pytest.raises(FileNotFoundError):
        load_benchmark_dataset(tmp_path / "inexistente.json")

    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text(json.dumps({"erro": "nao e lista"}), encoding="utf-8")
    with pytest.raises(ValueError, match="lista JSON"):
        load_benchmark_dataset(invalid_file)


def test_evaluate_single_query_calculates_precision_recall_and_mrr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valida o cálculo exato de Precision@k, Recall@k e Reciprocal Rank em consulta individual."""
    fake_retrieved = [
        {"title": "Doc Incorreto", "score": 0.75},
        {"title": "Doc Correto", "score": 0.70},
        {"title": "Outro Doc", "score": 0.60},
    ]
    monkeypatch.setattr(
        repository,
        "search_regulation_chunks",
        lambda query, top_k=3, **kw: fake_retrieved,
    )

    query_item = {
        "id": "q_test",
        "query": "consulta sobre norma",
        "category": "teste",
        "is_in_scope": True,
        "expected_relevant_titles": ["Doc Correto"],
    }

    result = evaluate_single_query(query_item, top_k=3, threshold=0.68)

    assert result.query_id == "q_test"
    assert result.precision_at_k[1] == 0.0
    assert result.precision_at_k[2] == 0.5  # 1 de 2 é correto
    assert result.recall_at_k[1] == 0.0
    assert result.recall_at_k[2] == 1.0  # O único esperado foi recuperado na posição 2
    assert result.reciprocal_rank == 0.5  # 1 / posição 2


def test_evaluate_single_query_identifies_out_of_scope_and_threshold_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Garante que consultas fora de escopo são avaliadas quanto a falsos positivos e filtragem por limiar."""
    # Cenário A: distrator com score alto (falso positivo não filtrado)
    monkeypatch.setattr(
        repository,
        "search_regulation_chunks",
        lambda query, top_k=3, **kw: [{"title": "Doc Ruido", "score": 0.75}],
    )
    res_high = evaluate_single_query(
        {"id": "q_out1", "query": "assunto fora de escopo", "is_in_scope": False, "expected_relevant_titles": []},
        top_k=1,
        threshold=0.68,
    )
    assert res_high.is_unsupported_candidate is True
    assert res_high.filtered_out_by_threshold is False

    # Cenário B: distrator com score baixo (corretamente filtrado pelo limiar)
    monkeypatch.setattr(
        repository,
        "search_regulation_chunks",
        lambda query, top_k=3, **kw: [{"title": "Doc Ruido", "score": 0.55}],
    )
    res_low = evaluate_single_query(
        {"id": "q_out2", "query": "assunto fora de escopo", "is_in_scope": False, "expected_relevant_titles": []},
        top_k=1,
        threshold=0.68,
    )
    assert res_low.is_unsupported_candidate is True
    assert res_low.filtered_out_by_threshold is True


def test_compute_benchmark_summary_aggregates_metrics_correctly() -> None:
    """Valida a consolidação das médias de Precision, Recall, MRR, taxas de respostas sem suporte e latência."""
    r1 = QueryEvaluationResult(
        query_id="q1",
        query="pergunta 1",
        category="cat1",
        is_in_scope=True,
        expected_titles=["T1"],
        retrieved_titles=["T1", "T2"],
        retrieved_scores=[0.8, 0.6],
        latency_ms=10.0,
        precision_at_k={1: 1.0, 2: 0.5},
        recall_at_k={1: 1.0, 2: 1.0},
        reciprocal_rank=1.0,
        is_unsupported_candidate=False,
        filtered_out_by_threshold=False,
    )
    r2 = QueryEvaluationResult(
        query_id="q2",
        query="pergunta 2 fora",
        category="fora",
        is_in_scope=False,
        expected_titles=[],
        retrieved_titles=["T2"],
        retrieved_scores=[0.55],
        latency_ms=20.0,
        precision_at_k={1: 0.0, 2: 0.0},
        recall_at_k={1: 0.0, 2: 0.0},
        reciprocal_rank=0.0,
        is_unsupported_candidate=True,
        filtered_out_by_threshold=True,  # Filtrado pelo limiar
    )

    summary = compute_benchmark_summary([r1, r2], top_k=2, threshold=0.68)

    assert summary.total_queries == 2
    assert summary.in_scope_queries == 1
    assert summary.out_of_scope_queries == 1
    assert summary.mean_precision_at_k[1] == 1.0
    assert summary.mean_recall_at_k[1] == 1.0
    assert summary.mrr == 1.0
    assert summary.latency_ms_mean == 15.0
    assert summary.unfiltered_unsupported_rate == 1.0  # 1 de 1 fora de escopo retornou ruído
    assert summary.filtered_unsupported_rate == 0.0  # Com threshold, foi filtrado


def test_format_markdown_report_generates_valid_table_content() -> None:
    """Garante que a geração do relatório Markdown contém cabeçalhos e seções técnicas esperadas."""
    summary = BenchmarkSummary(
        total_queries=1,
        in_scope_queries=1,
        out_of_scope_queries=0,
        top_k=3,
        threshold=0.68,
        mean_precision_at_k={1: 1.0, 3: 0.33},
        mean_recall_at_k={1: 1.0, 3: 1.0},
        mrr=1.0,
        latency_ms_mean=12.5,
        latency_ms_p50=12.5,
        latency_ms_p95=12.5,
        latency_ms_min=12.5,
        latency_ms_max=12.5,
        unfiltered_unsupported_rate=0.0,
        filtered_unsupported_rate=0.0,
    )
    report = format_markdown_report(summary, [])

    assert "# Relatorio de Avaliacao do RAG" in report
    assert "Precision@1" in report
    assert "MRR" in report
    assert "Latencia Media" in report


def test_run_rag_benchmark_end_to_end_in_test_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Valida a execução de ponta a ponta do benchmark sobre o dataset versionado com mock determinístico."""
    dataset_file = tmp_path / "mini_dataset.json"
    dataset_file.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "query": "qual o prazo do auditorio?",
                    "category": "auditorio",
                    "is_in_scope": True,
                    "expected_relevant_titles": ["Reserva de Auditorio"],
                },
                {
                    "id": "q2",
                    "query": "horario restaurante",
                    "category": "fora",
                    "is_in_scope": False,
                    "expected_relevant_titles": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    fake_results = {
        "qual o prazo do auditorio?": [
            {"title": "Reserva de Auditorio", "score": 0.85},
            {"title": "Outro", "score": 0.40},
        ],
        "horario restaurante": [
            {"title": "Outro", "score": 0.50},
        ],
    }

    monkeypatch.setattr(
        repository,
        "search_regulation_chunks",
        lambda query, top_k=3, **kw: fake_results.get(query, []),
    )

    summary, results = run_rag_benchmark(dataset_path=dataset_file, top_k=2, threshold=0.68)

    assert summary.total_queries == 2
    assert len(results) == 2
    assert summary.mean_precision_at_k[1] == 1.0
    assert summary.filtered_unsupported_rate == 0.0
