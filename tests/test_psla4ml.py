"""
Testes para gridsearch.tiers — Discretização PSLA4ML.

Coberturas:
  - compute_thresholds: estratégias median/mean/q1/q3, limiares explícitos,
    mistura de explícito + automático, resultado vazio, estratégia inválida
  - discretize_metrics: campos "discretized" e "discretization_thresholds",
    direção dos intervalos (< / ≥), referência externa, n/a para missing
  - _format_threshold: inteiros, floats pequenos, notação científica
  - _interval_for: limites de borda (igual ao limiar → "≥")
"""

import pytest
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(idx: int, time: float, energy: float, co2: float, cost: float,
          status: str = "success") -> Dict[str, Any]:
    return {
        "status": status,
        "grid_experiment_idx": idx,
        "grid_params": {"lr": 1e-5, "bs": 8},
        "resources": {
            "train_time_sec":   time,
            "energy_kwh":       energy,
            "emissions_kg_co2": co2,
            "cost_usd":         cost,
        },
    }


# Conjunto sintético com distribuição análoga ao artigo:
# GPU: tempo ~33 s, custo alto; CPU: tempo ~6000 s, custo baixo
@pytest.fixture
def traces_paper_like():
    """9 traces simulando 3 ambientes × 3 otimizadores (semelhante ao artigo)."""
    return [
        # GPU (rápido, caro)
        _make(0, 30.29, 0.001577, 0.000714, 1.21),
        _make(1, 32.10, 0.001652, 0.000748, 1.22),
        _make(2, 36.41, 0.001847, 0.000836, 1.20),
        # TPU (moderado, moderado)
        _make(3, 3634.15, 0.093357, 0.024984, 1.51),
        _make(4, 3662.99, 0.094093, 0.025181, 1.52),
        _make(5, 3701.62, 0.095077, 0.025445, 1.53),
        # CPU (lento, barato)
        _make(6, 5410.80, 0.210364, 0.056298, 0.55),
        _make(7, 6793.03, 0.264134, 0.070688, 0.68),
        _make(8, 6805.34, 0.264602, 0.070813, 0.69),
    ]


@pytest.fixture
def gpu_traces():
    """Apenas 3 traces de GPU para testar sem referência externa."""
    return [
        _make(0, 30.0, 0.001, 0.0007, 1.20),
        _make(1, 33.0, 0.002, 0.0008, 1.25),
        _make(2, 36.0, 0.003, 0.0009, 1.30),
    ]


# ---------------------------------------------------------------------------
# Importações públicas
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_from_gridsearch_tiers(self):
        from gridsearch.tiers import (
            compute_thresholds,
            discretize_metrics,
            DISCRETIZATION_STRATEGIES,
        )
        assert callable(compute_thresholds)
        assert callable(discretize_metrics)
        assert isinstance(DISCRETIZATION_STRATEGIES, tuple)

    def test_import_via_gridsearch_package(self):
        from gridsearch import compute_thresholds
        assert callable(compute_thresholds)

    def test_import_via_skyband_facade(self):
        from gridsearch.skyband import compute_thresholds
        assert callable(compute_thresholds)

    def test_discretization_strategies_values(self):
        from gridsearch.tiers import DISCRETIZATION_STRATEGIES
        assert "median" in DISCRETIZATION_STRATEGIES
        assert "mean" in DISCRETIZATION_STRATEGIES
        assert "q1" in DISCRETIZATION_STRATEGIES
        assert "q3" in DISCRETIZATION_STRATEGIES


# ---------------------------------------------------------------------------
# compute_thresholds
# ---------------------------------------------------------------------------

class TestComputeThresholds:
    def test_median_odd_n(self, gpu_traces):
        from gridsearch.tiers import compute_thresholds
        t = compute_thresholds(gpu_traces, ["train_time_sec"], strategy="median")
        # mediana de [30, 33, 36] = 33
        assert t["train_time_sec"] == pytest.approx(33.0)

    def test_median_even_n(self):
        from gridsearch.tiers import compute_thresholds
        traces = [_make(i, float(v), 0.1, 0.01, 1.0) for i, v in enumerate([10, 20, 30, 40])]
        t = compute_thresholds(traces, ["train_time_sec"], strategy="median")
        # mediana de [10, 20, 30, 40] = 25.0
        assert t["train_time_sec"] == pytest.approx(25.0)

    def test_mean_strategy(self, gpu_traces):
        from gridsearch.tiers import compute_thresholds
        t = compute_thresholds(gpu_traces, ["train_time_sec"], strategy="mean")
        assert t["train_time_sec"] == pytest.approx((30.0 + 33.0 + 36.0) / 3.0)

    def test_q1_strategy(self):
        from gridsearch.tiers import compute_thresholds
        # 4 valores: [10, 20, 30, 40] → q1 índice = (4-1)//4 = 0 → 10
        traces = [_make(i, float(v), 0.1, 0.01, 1.0) for i, v in enumerate([10, 20, 30, 40])]
        t = compute_thresholds(traces, ["train_time_sec"], strategy="q1")
        assert t["train_time_sec"] == pytest.approx(10.0)

    def test_q3_strategy(self):
        from gridsearch.tiers import compute_thresholds
        # 4 valores: [10, 20, 30, 40] → q3 índice = 3*(4-1)//4 = 2 → 30
        traces = [_make(i, float(v), 0.1, 0.01, 1.0) for i, v in enumerate([10, 20, 30, 40])]
        t = compute_thresholds(traces, ["train_time_sec"], strategy="q3")
        assert t["train_time_sec"] == pytest.approx(30.0)

    def test_explicit_threshold_overrides(self, traces_paper_like):
        from gridsearch.tiers import compute_thresholds
        t = compute_thresholds(
            traces_paper_like,
            ["train_time_sec"],
            explicit_thresholds={"train_time_sec": 5000.0},
        )
        assert t["train_time_sec"] == pytest.approx(5000.0)

    def test_mixed_explicit_and_auto(self, traces_paper_like):
        from gridsearch.tiers import compute_thresholds
        t = compute_thresholds(
            traces_paper_like,
            ["train_time_sec", "cost_usd"],
            strategy="median",
            explicit_thresholds={"cost_usd": 1.2},
        )
        # train_time_sec calculado automaticamente
        assert "train_time_sec" in t
        # cost_usd sobrescrito pelo explícito
        assert t["cost_usd"] == pytest.approx(1.2)

    def test_ignores_failed_results(self, traces_paper_like):
        from gridsearch.tiers import compute_thresholds
        traces_with_failed = traces_paper_like + [
            _make(99, 99999.0, 99.0, 99.0, 99.0, status="failed")
        ]
        t_with = compute_thresholds(traces_with_failed, ["train_time_sec"])
        t_without = compute_thresholds(traces_paper_like, ["train_time_sec"])
        # Resultado idêntico pois o trace failed é ignorado
        assert t_with["train_time_sec"] == pytest.approx(t_without["train_time_sec"])

    def test_empty_results_returns_empty(self):
        from gridsearch.tiers import compute_thresholds
        t = compute_thresholds([], ["train_time_sec"])
        assert t == {}

    def test_invalid_strategy_raises(self, gpu_traces):
        from gridsearch.tiers import compute_thresholds
        with pytest.raises(ValueError, match="Estratégia inválida"):
            compute_thresholds(gpu_traces, ["train_time_sec"], strategy="invalid")

    def test_multiple_metrics(self, traces_paper_like):
        from gridsearch.tiers import compute_thresholds
        metrics = ["train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"]
        t = compute_thresholds(traces_paper_like, metrics)
        assert set(t.keys()) == set(metrics)


# ---------------------------------------------------------------------------
# _format_threshold e _interval_for (internos, mas críticos para o artigo)
# ---------------------------------------------------------------------------

class TestFormatThreshold:
    def test_integer_large(self):
        from gridsearch.tiers import _format_threshold
        assert _format_threshold(5000.0) == "5000"

    def test_integer_small(self):
        from gridsearch.tiers import _format_threshold
        assert _format_threshold(1.0) == "1"

    def test_float_small(self):
        from gridsearch.tiers import _format_threshold
        result = _format_threshold(0.2)
        assert result == "0.2"

    def test_very_small_scientific(self):
        from gridsearch.tiers import _format_threshold
        result = _format_threshold(0.0005)
        assert "e" in result.lower() or float(result) == pytest.approx(0.0005, rel=1e-2)

    def test_zero(self):
        from gridsearch.tiers import _format_threshold
        assert _format_threshold(0.0) == "0"


class TestIntervalFor:
    def test_below_threshold(self):
        from gridsearch.tiers import _interval_for
        assert _interval_for(33.0, 5000.0) == "< 5000"

    def test_above_threshold(self):
        from gridsearch.tiers import _interval_for
        assert _interval_for(6000.0, 5000.0) == "≥ 5000"

    def test_equal_to_threshold_is_geq(self):
        from gridsearch.tiers import _interval_for
        # Valor exatamente igual ao limiar → intervalo "≥"
        assert _interval_for(5000.0, 5000.0) == "≥ 5000"

    def test_paper_gpu_time(self):
        from gridsearch.tiers import _interval_for
        assert _interval_for(33.0, 5000.0) == "< 5000"

    def test_paper_cpu_time(self):
        from gridsearch.tiers import _interval_for
        assert _interval_for(6336.0, 5000.0) == "≥ 5000"


# ---------------------------------------------------------------------------
# discretize_metrics
# ---------------------------------------------------------------------------

class TestDiscreteizeMetrics:
    def test_adds_discretized_key(self, gpu_traces):
        from gridsearch.tiers import discretize_metrics
        result = discretize_metrics(gpu_traces, metrics=["train_time_sec"])
        for r in result:
            assert "discretized" in r
            assert "train_time_sec" in r["discretized"]

    def test_adds_thresholds_key(self, gpu_traces):
        from gridsearch.tiers import discretize_metrics
        result = discretize_metrics(gpu_traces, metrics=["train_time_sec"])
        for r in result:
            assert "discretization_thresholds" in r
            assert "train_time_sec" in r["discretization_thresholds"]

    def test_does_not_mutate_input(self, gpu_traces):
        from gridsearch.tiers import discretize_metrics
        originals = [dict(r) for r in gpu_traces]
        discretize_metrics(gpu_traces, metrics=["train_time_sec"])
        for orig, current in zip(originals, gpu_traces):
            assert "discretized" not in current
            assert orig == current

    def test_interval_below_median(self):
        from gridsearch.tiers import discretize_metrics
        traces = [_make(i, float(v), 0.1, 0.01, 1.0) for i, v in enumerate([10, 20, 30])]
        # mediana = 20; valor 10 → "< 20"
        result = discretize_metrics(traces, metrics=["train_time_sec"])
        assert result[0]["discretized"]["train_time_sec"] == "< 20"

    def test_interval_above_median(self):
        from gridsearch.tiers import discretize_metrics
        traces = [_make(i, float(v), 0.1, 0.01, 1.0) for i, v in enumerate([10, 20, 30])]
        # mediana = 20; valor 30 → "≥ 20"
        result = discretize_metrics(traces, metrics=["train_time_sec"])
        assert result[2]["discretized"]["train_time_sec"] == "≥ 20"

    def test_reference_results_used_for_thresholds(self, traces_paper_like, gpu_traces):
        from gridsearch.tiers import discretize_metrics
        # Discretiza apenas os gpu_traces, mas usa traces_paper_like como referência
        result = discretize_metrics(
            results=gpu_traces,
            metrics=["train_time_sec"],
            reference_results=traces_paper_like,
        )
        # Limiar calculado sobre traces_paper_like (mediana ≈ 3634s)
        # → todos os gpu_traces (~33s) ficam abaixo → "< ..."
        for r in result:
            assert r["discretized"]["train_time_sec"].startswith("< ")

    def test_explicit_thresholds_override(self, traces_paper_like, gpu_traces):
        from gridsearch.tiers import discretize_metrics
        result = discretize_metrics(
            results=gpu_traces,
            metrics=["train_time_sec"],
            thresholds={"train_time_sec": 5000.0},
        )
        for r in result:
            assert r["discretized"]["train_time_sec"] == "< 5000"

    def test_missing_metric_returns_na(self):
        from gridsearch.tiers import discretize_metrics
        trace = {"status": "success", "grid_experiment_idx": 0,
                 "grid_params": {}, "resources": {}}
        result = discretize_metrics([trace], metrics=["train_time_sec"])
        # Sem valor na resources → n/a
        assert result[0]["discretized"]["train_time_sec"] == "n/a"

    def test_empty_results_returns_empty(self):
        from gridsearch.tiers import discretize_metrics
        assert discretize_metrics([]) == []

    def test_default_metrics_when_none(self, gpu_traces):
        from gridsearch.tiers import discretize_metrics
        result = discretize_metrics(gpu_traces)
        # Apenas métricas presentes nos traces são discretizadas (total_gflops ausente)
        for r in result:
            assert "discretized" in r

    def test_returns_same_length_as_input(self, traces_paper_like):
        from gridsearch.tiers import discretize_metrics
        result = discretize_metrics(
            traces_paper_like,
            metrics=["train_time_sec", "cost_usd"],
        )
        assert len(result) == len(traces_paper_like)


# ---------------------------------------------------------------------------
# Reprodução dos valores do artigo (Tabela 3)
# ---------------------------------------------------------------------------

class TestPaperTable3Reproduction:
    """Verifica que discretize_metrics reproduz a Tabela 3 do artigo PSLA4ML
    quando os limiares explícitos do paper são fornecidos."""

    PAPER_THRESHOLDS = {
        "train_time_sec":   5000.0,
        "energy_kwh":       0.2,
        "emissions_kg_co2": 0.05,
        "cost_usd":         1.2,
    }
    METRICS = list(PAPER_THRESHOLDS.keys())

    def _gpu_trace(self):
        """k=1 GPU trace do artigo: tempo ~33s, energia ~0.002, CO2 ~0.0007, custo ~1.21."""
        return _make(0, 30.29, 0.001577, 0.000714, 1.21)

    def _cpu_trace(self):
        """k=1 CPU trace do artigo: tempo ~6793s, energia ~0.264, CO2 ~0.071, custo ~0.68."""
        return _make(1, 6793.03, 0.264134, 0.070688, 0.68)

    def test_gpu_trace_k1(self):
        from gridsearch.tiers import discretize_metrics
        result = discretize_metrics(
            [self._gpu_trace()],
            metrics=self.METRICS,
            thresholds=self.PAPER_THRESHOLDS,
        )
        disc = result[0]["discretized"]
        assert disc["train_time_sec"]   == "< 5000"
        assert disc["energy_kwh"]       == "< 0.2"
        assert disc["emissions_kg_co2"] == "< 0.05"
        assert disc["cost_usd"]         == "≥ 1.2"

    def test_cpu_trace_k1(self):
        from gridsearch.tiers import discretize_metrics
        result = discretize_metrics(
            [self._cpu_trace()],
            metrics=self.METRICS,
            thresholds=self.PAPER_THRESHOLDS,
        )
        disc = result[0]["discretized"]
        assert disc["train_time_sec"]   == "≥ 5000"
        assert disc["energy_kwh"]       == "≥ 0.2"
        assert disc["emissions_kg_co2"] == "≥ 0.05"
        assert disc["cost_usd"]         == "< 1.2"

    def test_all_four_metrics_present(self):
        from gridsearch.tiers import discretize_metrics
        result = discretize_metrics(
            [self._gpu_trace()],
            metrics=self.METRICS,
            thresholds=self.PAPER_THRESHOLDS,
        )
        assert set(result[0]["discretized"].keys()) == set(self.METRICS)
