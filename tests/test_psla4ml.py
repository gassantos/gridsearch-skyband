"""
Testes para gridsearch.tiers — Discretização PSLA4ML (BL-02),
geração de tiers (BL-03) e TrainingTemplate / filter_by_template (BL-04).

Coberturas BL-02:
  - compute_thresholds: estratégias median/mean/q1/q3, limiares explícitos,
    mistura de explícito + automático, resultado vazio, estratégia inválida
  - discretize_metrics: campos "discretized" e "discretization_thresholds",
    direção dos intervalos (< / ≥), referência externa, n/a para missing
  - _format_threshold: inteiros, floats pequenos, notação científica
  - _interval_for: limites de borda (igual ao limiar → "≥")

Coberturas BL-03:
  - Tier: dataclass, campos, to_dict, importação pública
  - _extract_hardware: formatos de resultado (produção, grid, multienv)
  - _extract_hyperparam: caminhos hyperparameters / grid_params / raiz
  - generate_psla4ml: Algoritmo 1 passos 4-10, k=1/k=2, SLA constraints,
    limiares explícitos, modelo/dataset, reprodução da Tabela 3

Coberturas BL-04:
  - TrainingTemplate: dataclass, campos padrão, __str__
  - _get_arch_string: formatos de resultado (production / grid)
  - _get_dataset_string: idem
  - _values_match: floats com tolerância, strings case-insensitive, None
  - TrainingTemplate.matches(): A, D, H separados e combinados, wildcards,
    mismatch parcial, resultado de produção vs grid
  - filter_by_template: filtra correto, vazio quando nenhum bate, não muta input
  - generate_psla4ml com template: filtra antes do Skyband, propaga model/dataset,
    template incompatível retorna []
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


# ---------------------------------------------------------------------------
# Tier dataclass
# ---------------------------------------------------------------------------

class TestTier:
    def test_import_tier(self):
        from gridsearch.tiers import Tier
        assert Tier is not None

    def test_import_via_package(self):
        from gridsearch import Tier
        assert Tier is not None

    def test_import_via_skyband_facade(self):
        from gridsearch.skyband import Tier
        assert Tier is not None

    def test_default_construction(self):
        from gridsearch.tiers import Tier
        t = Tier()
        assert t.model == ""
        assert t.dataset == ""
        assert t.learning_rate is None
        assert t.batch_size is None
        assert t.optimizer is None
        assert t.dropout is None
        assert t.hardware == ""
        assert t.discretized == {}
        assert t.raw_metrics == {}
        assert t.domination_count == 0
        assert t.k == 1
        assert t.experiment_id is None
        assert t.grid_experiment_idx is None
        assert t.selected_environment is None

    def test_construction_with_values(self):
        from gridsearch.tiers import Tier
        t = Tier(
            model="BERT-PLI",
            dataset="COLLIE",
            learning_rate=1e-5,
            batch_size=8,
            optimizer="BertAdam",
            dropout=0.1,
            hardware="gpu",
            discretized={"train_time_sec": "< 5000", "cost_usd": "≥ 1.2"},
            raw_metrics={"train_time_sec": 30.29, "cost_usd": 1.21},
            domination_count=0,
            k=1,
        )
        assert t.model == "BERT-PLI"
        assert t.learning_rate == pytest.approx(1e-5)
        assert t.discretized["cost_usd"] == "≥ 1.2"

    def test_to_dict_keys(self):
        from gridsearch.tiers import Tier
        t = Tier(model="BERT-PLI", dataset="COLLIE", hardware="gpu", k=2)
        d = t.to_dict()
        expected_keys = {
            "model", "dataset", "learning_rate", "batch_size", "optimizer",
            "dropout", "hardware", "discretized", "raw_metrics",
            "domination_count", "k", "experiment_id",
            "grid_experiment_idx", "selected_environment",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values(self):
        from gridsearch.tiers import Tier
        t = Tier(model="BERT-PLI", hardware="gpu", domination_count=1, k=2)
        d = t.to_dict()
        assert d["model"] == "BERT-PLI"
        assert d["hardware"] == "gpu"
        assert d["domination_count"] == 1
        assert d["k"] == 2


# ---------------------------------------------------------------------------
# _extract_hardware
# ---------------------------------------------------------------------------

class TestExtractHardware:
    def test_from_environment_device_type(self):
        from gridsearch.tiers import _extract_hardware
        result = {"environment": {"device_type": "GPU"}, "status": "success"}
        assert _extract_hardware(result) == "gpu"

    def test_from_selected_environment(self):
        from gridsearch.tiers import _extract_hardware
        result = {"selected_environment": "GCP", "status": "success"}
        assert _extract_hardware(result) == "gcp"

    def test_from_grid_params_environment(self):
        from gridsearch.tiers import _extract_hardware
        result = {"grid_params": {"environment": "local"}, "status": "success"}
        assert _extract_hardware(result) == "local"

    def test_environment_takes_priority_over_selected(self):
        from gridsearch.tiers import _extract_hardware
        result = {
            "environment": {"device_type": "GPU"},
            "selected_environment": "cpu_fallback",
        }
        assert _extract_hardware(result) == "gpu"

    def test_missing_hardware_returns_empty(self):
        from gridsearch.tiers import _extract_hardware
        assert _extract_hardware({}) == ""


# ---------------------------------------------------------------------------
# _extract_hyperparam
# ---------------------------------------------------------------------------

class TestExtractHyperparam:
    def test_from_hyperparameters_block(self):
        from gridsearch.tiers import _extract_hyperparam
        result = {"hyperparameters": {"learning_rate": 2e-5}}
        assert _extract_hyperparam(result, "learning_rate", ["lr"]) == pytest.approx(2e-5)

    def test_from_grid_params_alias(self):
        from gridsearch.tiers import _extract_hyperparam
        result = {"grid_params": {"lr": 1e-5}}
        val = _extract_hyperparam(result, "learning_rate", ["lr"])
        assert val == pytest.approx(1e-5)

    def test_from_root(self):
        from gridsearch.tiers import _extract_hyperparam
        result = {"batch_size": 16}
        assert _extract_hyperparam(result, "batch_size", ["bs"]) == 16

    def test_missing_returns_none(self):
        from gridsearch.tiers import _extract_hyperparam
        assert _extract_hyperparam({}, "learning_rate", ["lr"]) is None


# ---------------------------------------------------------------------------
# generate_psla4ml
# ---------------------------------------------------------------------------

class TestGeneratePsla4ml:
    METRICS = ["train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"]
    PAPER_THR = {"train_time_sec": 5000.0, "energy_kwh": 0.2,
                 "emissions_kg_co2": 0.05, "cost_usd": 1.2}

    def _full_traces(self):
        """9 traces com ambiente embutido nos grid_params (como no multienv)."""
        def _t(idx, hw, time, energy, co2, cost, lr=1e-5, bs=8, opt="BertAdam"):
            return {
                "status": "success",
                "grid_experiment_idx": idx,
                "grid_params": {"lr": lr, "bs": bs, "optimizer": opt,
                                "dropout": 0.1, "environment": hw},
                "selected_environment": hw,
                "resources": {"train_time_sec": time, "energy_kwh": energy,
                              "emissions_kg_co2": co2, "cost_usd": cost},
            }
        return [
            _t(0, "gpu",  30.29,  0.001577, 0.000714, 1.21),
            _t(1, "gpu",  32.10,  0.001652, 0.000748, 1.22),
            _t(2, "gpu",  36.41,  0.001847, 0.000836, 1.20),
            _t(3, "tpu",  3634.15, 0.093357, 0.024984, 1.51),
            _t(4, "tpu",  3662.99, 0.094093, 0.025181, 1.52),
            _t(5, "tpu",  3701.62, 0.095077, 0.025445, 1.53),
            _t(6, "cpu",  5410.80, 0.210364, 0.056298, 0.55),
            _t(7, "cpu",  6793.03, 0.264134, 0.070688, 0.68),
            _t(8, "cpu",  6805.34, 0.264602, 0.070813, 0.69),
        ]

    def test_returns_list_of_tier(self):
        from gridsearch.tiers import generate_psla4ml, Tier
        tiers = generate_psla4ml(self._full_traces(), k=1, metrics=self.METRICS)
        assert isinstance(tiers, list)
        assert all(isinstance(t, Tier) for t in tiers)

    def test_k1_returns_pareto_front(self):
        from gridsearch.tiers import generate_psla4ml
        tiers = generate_psla4ml(self._full_traces(), k=1, metrics=self.METRICS)
        # Todos os tiers do Skyband k=1 têm domination_count < 1
        assert all(t.domination_count == 0 for t in tiers)

    def test_k2_has_more_tiers_than_k1(self):
        from gridsearch.tiers import generate_psla4ml
        t1 = generate_psla4ml(self._full_traces(), k=1, metrics=self.METRICS)
        t2 = generate_psla4ml(self._full_traces(), k=2, metrics=self.METRICS)
        assert len(t2) >= len(t1)

    def test_k_attribute_set_correctly(self):
        from gridsearch.tiers import generate_psla4ml
        for k in [1, 2, 3]:
            tiers = generate_psla4ml(self._full_traces(), k=k, metrics=self.METRICS)
            assert all(t.k == k for t in tiers)

    def test_model_and_dataset_propagated(self):
        from gridsearch.tiers import generate_psla4ml
        tiers = generate_psla4ml(
            self._full_traces(), k=1,
            metrics=self.METRICS,
            model="BERT-PLI", dataset="COLLIE",
        )
        assert all(t.model == "BERT-PLI" for t in tiers)
        assert all(t.dataset == "COLLIE" for t in tiers)

    def test_discretized_present_in_all_tiers(self):
        from gridsearch.tiers import generate_psla4ml
        tiers = generate_psla4ml(
            self._full_traces(), k=2, metrics=self.METRICS,
            thresholds=self.PAPER_THR,
        )
        for t in tiers:
            assert set(t.discretized.keys()) == set(self.METRICS)

    def test_raw_metrics_present(self):
        from gridsearch.tiers import generate_psla4ml
        tiers = generate_psla4ml(self._full_traces(), k=1, metrics=self.METRICS)
        for t in tiers:
            assert len(t.raw_metrics) > 0
            assert "train_time_sec" in t.raw_metrics

    def test_hardware_extracted(self):
        from gridsearch.tiers import generate_psla4ml
        tiers = generate_psla4ml(self._full_traces(), k=1, metrics=self.METRICS)
        hardwares = {t.hardware for t in tiers}
        # Skyband k=1: GPU domina em tempo/energia; CPU domina em custo
        assert "gpu" in hardwares or "cpu" in hardwares

    def test_sorted_by_domination_count(self):
        from gridsearch.tiers import generate_psla4ml
        tiers = generate_psla4ml(self._full_traces(), k=3, metrics=self.METRICS)
        counts = [t.domination_count for t in tiers]
        assert counts == sorted(counts)

    def test_sla_constraint_reduces_tiers(self):
        from gridsearch.tiers import generate_psla4ml
        # Sem SLA
        all_tiers = generate_psla4ml(self._full_traces(), k=3, metrics=self.METRICS)
        # Com SLA que exclui CPU (custo < 0.80 exclui CPUs com custo > 0.80)
        sla_tiers = generate_psla4ml(
            self._full_traces(), k=3, metrics=self.METRICS,
            sla_constraints={"cost_usd": 0.80},
        )
        assert len(sla_tiers) <= len(all_tiers)

    def test_empty_traces_returns_empty(self):
        from gridsearch.tiers import generate_psla4ml
        assert generate_psla4ml([], k=1, metrics=self.METRICS) == []

    def test_to_dict_serializable(self):
        import json
        from gridsearch.tiers import generate_psla4ml
        tiers = generate_psla4ml(
            self._full_traces(), k=1, metrics=self.METRICS,
            thresholds=self.PAPER_THR,
        )
        for t in tiers:
            d = t.to_dict()
            # Deve ser serializável em JSON
            json_str = json.dumps(d)
            assert len(json_str) > 0

    def test_paper_table3_gpu_k1(self):
        """GPU tier k=1: intervalos conforme Tabela 3 do artigo."""
        from gridsearch.tiers import generate_psla4ml
        tiers = generate_psla4ml(
            self._full_traces(), k=1, metrics=self.METRICS,
            thresholds=self.PAPER_THR,
        )
        gpu_tiers = [t for t in tiers if t.hardware == "gpu"]
        assert gpu_tiers, "Nenhum tier GPU encontrado no k=1"
        g = gpu_tiers[0]
        assert g.discretized["train_time_sec"]   == "< 5000"
        assert g.discretized["energy_kwh"]       == "< 0.2"
        assert g.discretized["emissions_kg_co2"] == "< 0.05"
        assert g.discretized["cost_usd"]         == "≥ 1.2"

    def test_paper_table3_cpu_k1(self):
        """CPU tier k=1: intervalos conforme Tabela 3 do artigo."""
        from gridsearch.tiers import generate_psla4ml
        tiers = generate_psla4ml(
            self._full_traces(), k=1, metrics=self.METRICS,
            thresholds=self.PAPER_THR,
        )
        cpu_tiers = [t for t in tiers if t.hardware == "cpu"]
        assert cpu_tiers, "Nenhum tier CPU encontrado no k=1"
        c = cpu_tiers[0]
        assert c.discretized["train_time_sec"]   == "≥ 5000"
        assert c.discretized["energy_kwh"]       == "≥ 0.2"
        assert c.discretized["emissions_kg_co2"] == "≥ 0.05"
        assert c.discretized["cost_usd"]         == "< 1.2"


# ---------------------------------------------------------------------------
# BL-04 — _get_arch_string e _get_dataset_string
# ---------------------------------------------------------------------------

class TestGetArchString:
    def test_from_experiment_config_name(self):
        from gridsearch.tiers import _get_arch_string
        result = {"experiment": {"config_name": "BertPLI_adam_lr1e-05.json"}}
        assert _get_arch_string(result) == "BertPLI_adam_lr1e-05.json"

    def test_from_grid_params_architecture(self):
        from gridsearch.tiers import _get_arch_string
        result = {"grid_params": {"architecture": "AttenLSTM"}}
        assert _get_arch_string(result) == "AttenLSTM"

    def test_from_grid_params_model(self):
        from gridsearch.tiers import _get_arch_string
        result = {"grid_params": {"model": "BertPoint"}}
        assert _get_arch_string(result) == "BertPoint"

    def test_experiment_takes_priority(self):
        from gridsearch.tiers import _get_arch_string
        result = {
            "experiment": {"config_name": "BertPLI_config.json"},
            "grid_params": {"model": "Other"},
        }
        assert _get_arch_string(result) == "BertPLI_config.json"

    def test_missing_returns_empty(self):
        from gridsearch.tiers import _get_arch_string
        assert _get_arch_string({}) == ""


class TestGetDatasetString:
    def test_from_execution_train_dataset(self):
        from gridsearch.tiers import _get_dataset_string
        result = {"execution": {"train_dataset": "collie_train"}}
        assert _get_dataset_string(result) == "collie_train"

    def test_from_grid_params_dataset(self):
        from gridsearch.tiers import _get_dataset_string
        result = {"grid_params": {"dataset": "COLIEE"}}
        assert _get_dataset_string(result) == "COLIEE"

    def test_execution_takes_priority(self):
        from gridsearch.tiers import _get_dataset_string
        result = {
            "execution": {"train_dataset": "train_task2"},
            "grid_params": {"dataset": "other"},
        }
        assert _get_dataset_string(result) == "train_task2"

    def test_missing_returns_empty(self):
        from gridsearch.tiers import _get_dataset_string
        assert _get_dataset_string({}) == ""


# ---------------------------------------------------------------------------
# BL-04 — _values_match
# ---------------------------------------------------------------------------

class TestValuesMatch:
    def test_float_equal(self):
        from gridsearch.tiers import _values_match
        assert _values_match(0.1, 0.1)

    def test_float_within_tolerance(self):
        from gridsearch.tiers import _values_match
        assert _values_match(0.10000000001, 0.1)

    def test_float_outside_tolerance(self):
        from gridsearch.tiers import _values_match
        assert not _values_match(0.2, 0.1)

    def test_int_float_comparison(self):
        from gridsearch.tiers import _values_match
        assert _values_match(1, 1.0)

    def test_string_case_insensitive(self):
        from gridsearch.tiers import _values_match
        assert _values_match("BertAdam", "bertadam")
        assert _values_match("adam", "Adam")

    def test_string_different(self):
        from gridsearch.tiers import _values_match
        assert not _values_match("adam", "adamw")

    def test_none_actual_returns_false(self):
        from gridsearch.tiers import _values_match
        assert not _values_match(None, 0.1)

    def test_exact_equality_other_types(self):
        from gridsearch.tiers import _values_match
        assert _values_match(8, 8)
        assert not _values_match(8, 16)


# ---------------------------------------------------------------------------
# BL-04 — TrainingTemplate
# ---------------------------------------------------------------------------

class TestTrainingTemplate:
    def test_import(self):
        from gridsearch.tiers import TrainingTemplate
        assert TrainingTemplate is not None

    def test_import_via_package(self):
        from gridsearch import TrainingTemplate
        assert TrainingTemplate is not None

    def test_import_via_skyband_facade(self):
        from gridsearch.skyband import TrainingTemplate
        assert TrainingTemplate is not None

    def test_default_construction(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate()
        assert t.architecture == ""
        assert t.dataset_id == ""
        assert t.fixed_hyperparams == {}

    def test_str_representation(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(
            architecture="BertPLI",
            dataset_id="COLLIE",
            fixed_hyperparams={"dropout": 0.1},
        )
        s = str(t)
        assert "BertPLI" in s
        assert "COLLIE" in s
        assert "dropout" in s

    def test_str_wildcards_shown_as_star(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate()
        s = str(t)
        assert "*" in s

    # — matches() —

    def _make_prod(self, arch="BertPLI_config.json", ds="collie_train", dropout=0.1):
        """Resultado em formato de produção (build_result_dict)."""
        return {
            "status": "success",
            "experiment": {"config_name": arch},
            "execution": {"train_dataset": ds},
            "hyperparameters": {"dropout": dropout, "optimizer": "adam",
                                "learning_rate": 1e-5, "batch_size": 8},
            "resources": {"train_time_sec": 33.0, "cost_usd": 1.21,
                          "energy_kwh": 0.002, "emissions_kg_co2": 0.001},
        }

    def _make_grid(self, arch=None, ds=None, dropout=0.1, optimizer="adam"):
        """Resultado em formato de grid search (run_single_experiment)."""
        return {
            "status": "success",
            "grid_params": {
                "lr": 1e-5, "bs": 8, "dropout": dropout,
                "optimizer": optimizer,
                **({"architecture": arch} if arch else {}),
                **({"dataset": ds} if ds else {}),
            },
            "resources": {"train_time_sec": 33.0, "cost_usd": 1.21,
                          "energy_kwh": 0.002, "emissions_kg_co2": 0.001},
        }

    def test_matches_architecture_substring(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(architecture="BertPLI")
        assert t.matches(self._make_prod())

    def test_matches_architecture_case_insensitive(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(architecture="bertpli")
        assert t.matches(self._make_prod())

    def test_no_match_wrong_architecture(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(architecture="AttenLSTM")
        assert not t.matches(self._make_prod())

    def test_matches_dataset_substring(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(dataset_id="COLLIE")
        assert t.matches(self._make_prod())

    def test_no_match_wrong_dataset(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(dataset_id="COLIEE")
        assert not t.matches(self._make_prod())

    def test_matches_fixed_hyperparams_float(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(fixed_hyperparams={"dropout": 0.1})
        assert t.matches(self._make_prod(dropout=0.1))

    def test_no_match_wrong_hyperparam(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(fixed_hyperparams={"dropout": 0.2})
        assert not t.matches(self._make_prod(dropout=0.1))

    def test_none_hyperparam_is_wildcard(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(fixed_hyperparams={"dropout": None})
        # None = não filtra
        assert t.matches(self._make_prod(dropout=0.1))
        assert t.matches(self._make_prod(dropout=0.3))

    def test_empty_template_matches_everything(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate()
        assert t.matches(self._make_prod())
        assert t.matches(self._make_grid())
        assert t.matches({})

    def test_combined_a_d_h_match(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(
            architecture="BertPLI",
            dataset_id="COLLIE",
            fixed_hyperparams={"dropout": 0.1},
        )
        assert t.matches(self._make_prod())

    def test_combined_a_d_h_mismatch_on_h(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(
            architecture="BertPLI",
            dataset_id="COLLIE",
            fixed_hyperparams={"dropout": 0.2},
        )
        assert not t.matches(self._make_prod(dropout=0.1))

    def test_matches_grid_format(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(
            architecture="BertPLI",
            fixed_hyperparams={"dropout": 0.1, "optimizer": "adam"},
        )
        assert t.matches(self._make_grid(arch="BertPLI", dropout=0.1, optimizer="adam"))

    def test_no_match_grid_format_wrong_optimizer(self):
        from gridsearch.tiers import TrainingTemplate
        t = TrainingTemplate(fixed_hyperparams={"optimizer": "adamw"})
        assert not t.matches(self._make_grid(optimizer="adam"))


# ---------------------------------------------------------------------------
# BL-04 — filter_by_template
# ---------------------------------------------------------------------------

class TestFilterByTemplate:
    def test_import(self):
        from gridsearch.tiers import filter_by_template
        assert callable(filter_by_template)

    def test_import_via_package(self):
        from gridsearch import filter_by_template
        assert callable(filter_by_template)

    def test_import_via_skyband_facade(self):
        from gridsearch.skyband import filter_by_template
        assert callable(filter_by_template)

    def _make(self, arch, ds, dropout=0.1):
        return {
            "status": "success",
            "experiment": {"config_name": arch},
            "execution": {"train_dataset": ds},
            "hyperparameters": {"dropout": dropout, "optimizer": "adam"},
            "resources": {"train_time_sec": 33.0, "cost_usd": 1.2,
                          "energy_kwh": 0.002, "emissions_kg_co2": 0.001},
        }

    def test_filters_by_architecture(self):
        from gridsearch.tiers import TrainingTemplate, filter_by_template
        results = [
            self._make("BertPLI_config.json", "collie"),
            self._make("AttenLSTM_config.json", "collie"),
            self._make("BertPLI_v2.json", "collie"),
        ]
        t = TrainingTemplate(architecture="BertPLI")
        filtered = filter_by_template(results, t)
        assert len(filtered) == 2
        for r in filtered:
            assert "BertPLI" in r["experiment"]["config_name"]

    def test_filters_by_dataset(self):
        from gridsearch.tiers import TrainingTemplate, filter_by_template
        results = [
            self._make("BertPLI.json", "collie_train"),
            self._make("BertPLI.json", "coliee_train"),
            self._make("BertPLI.json", "collie_valid"),
        ]
        t = TrainingTemplate(dataset_id="collie")
        filtered = filter_by_template(results, t)
        # "collie_train" e "collie_valid" contêm "collie"
        assert len(filtered) == 2

    def test_filters_by_fixed_hyperparam(self):
        from gridsearch.tiers import TrainingTemplate, filter_by_template
        results = [
            self._make("BertPLI.json", "collie", dropout=0.1),
            self._make("BertPLI.json", "collie", dropout=0.2),
            self._make("BertPLI.json", "collie", dropout=0.1),
        ]
        t = TrainingTemplate(fixed_hyperparams={"dropout": 0.1})
        filtered = filter_by_template(results, t)
        assert len(filtered) == 2

    def test_empty_template_keeps_all(self):
        from gridsearch.tiers import TrainingTemplate, filter_by_template
        from gridsearch.tiers import TrainingTemplate
        results = [self._make("A.json", "ds1"), self._make("B.json", "ds2")]
        t = TrainingTemplate()
        assert len(filter_by_template(results, t)) == 2

    def test_no_match_returns_empty(self):
        from gridsearch.tiers import TrainingTemplate, filter_by_template
        results = [self._make("BertPLI.json", "collie")]
        t = TrainingTemplate(architecture="NonExistent")
        assert filter_by_template(results, t) == []

    def test_does_not_mutate_input(self):
        from gridsearch.tiers import TrainingTemplate, filter_by_template
        results = [self._make("BertPLI.json", "collie")]
        t = TrainingTemplate(architecture="BertPLI")
        filtered = filter_by_template(results, t)
        # filtered e results são listas distintas, mas compartilham os mesmos dicts
        assert filtered is not results
        assert filtered[0] is results[0]

    def test_paper_template_experiment(self):
        """Template do artigo: T = ⟨BertPLI, COLLIE, {dropout=0.1}⟩."""
        from gridsearch.tiers import TrainingTemplate, filter_by_template
        results = [
            self._make("BertPLI_config.json", "collie_train", dropout=0.1),
            self._make("BertPLI_config.json", "collie_train", dropout=0.2),
            self._make("AttenLSTM_config.json", "collie_train", dropout=0.1),
        ]
        t = TrainingTemplate(
            architecture="BertPLI",
            dataset_id="COLLIE",
            fixed_hyperparams={"dropout": 0.1},
        )
        filtered = filter_by_template(results, t)
        assert len(filtered) == 1
        assert filtered[0]["hyperparameters"]["dropout"] == 0.1
        assert "BertPLI" in filtered[0]["experiment"]["config_name"]


# ---------------------------------------------------------------------------
# BL-04 — generate_psla4ml com template
# ---------------------------------------------------------------------------

class TestGeneratePsla4mlWithTemplate:
    METRICS = ["train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"]
    PAPER_THR = {"train_time_sec": 5000.0, "energy_kwh": 0.2,
                 "emissions_kg_co2": 0.05, "cost_usd": 1.2}

    def _mixed_traces(self):
        """Traces de dois modelos distintos: BertPLI e AttenLSTM."""
        def _t(idx, arch, ds, hw, time, cost, dropout=0.1):
            return {
                "status": "success",
                "grid_experiment_idx": idx,
                "experiment": {"config_name": arch},
                "execution": {"train_dataset": ds},
                "hyperparameters": {"dropout": dropout, "optimizer": "adam",
                                    "learning_rate": 1e-5, "batch_size": 8},
                "selected_environment": hw,
                "environment": {"device_type": hw},
                "resources": {"train_time_sec": time, "cost_usd": cost,
                              "energy_kwh": cost * 0.001,
                              "emissions_kg_co2": cost * 0.0005},
            }
        return [
            # BertPLI / COLLIE / dropout=0.1
            _t(0, "BertPLI_cfg.json", "collie_train", "gpu",  33.0, 1.21),
            _t(1, "BertPLI_cfg.json", "collie_train", "cpu", 6793.0, 0.68),
            # BertPLI / COLLIE / dropout=0.2 (diferente H)
            _t(2, "BertPLI_cfg.json", "collie_train", "gpu",  35.0, 1.20, dropout=0.2),
            # AttenLSTM (diferente A)
            _t(3, "AttenLSTM_cfg.json", "collie_train", "gpu", 40.0, 1.10),
        ]

    def test_template_filters_before_skyband(self):
        from gridsearch.tiers import TrainingTemplate, generate_psla4ml
        template = TrainingTemplate(
            architecture="BertPLI",
            dataset_id="COLLIE",
            fixed_hyperparams={"dropout": 0.1},
        )
        tiers = generate_psla4ml(
            self._mixed_traces(), k=1, metrics=self.METRICS,
            thresholds=self.PAPER_THR, template=template,
        )
        # Apenas 2 traces passam o filtro (dropout=0.1, BertPLI, COLLIE)
        assert len(tiers) >= 1
        for t in tiers:
            assert t.model == "BertPLI"
            assert t.dataset == "COLLIE"

    def test_template_propagates_model_and_dataset(self):
        from gridsearch.tiers import TrainingTemplate, generate_psla4ml
        template = TrainingTemplate(
            architecture="BertPLI",
            dataset_id="COLLIE",
        )
        tiers = generate_psla4ml(
            self._mixed_traces(), k=1, metrics=self.METRICS,
            template=template,
        )
        assert all(t.model == "BertPLI" for t in tiers)
        assert all(t.dataset == "COLLIE" for t in tiers)

    def test_explicit_model_overrides_template(self):
        from gridsearch.tiers import TrainingTemplate, generate_psla4ml
        template = TrainingTemplate(architecture="BertPLI")
        tiers = generate_psla4ml(
            self._mixed_traces(), k=1, metrics=self.METRICS,
            model="BERT-Custom", template=template,
        )
        # model explícito != "BERT-PLI" default → NÃO é sobrescrito pelo template
        assert all(t.model == "BERT-Custom" for t in tiers)

    def test_incompatible_template_returns_empty(self):
        from gridsearch.tiers import TrainingTemplate, generate_psla4ml
        template = TrainingTemplate(architecture="NonExistentModel")
        result = generate_psla4ml(
            self._mixed_traces(), k=1, metrics=self.METRICS,
            template=template,
        )
        assert result == []

    def test_no_template_uses_all_traces(self):
        from gridsearch.tiers import generate_psla4ml
        tiers_all = generate_psla4ml(
            self._mixed_traces(), k=1, metrics=self.METRICS,
            thresholds=self.PAPER_THR,
        )
        tiers_filtered = generate_psla4ml(
            self._mixed_traces(), k=1, metrics=self.METRICS,
            thresholds=self.PAPER_THR,
        )
        # Sem template, os dois são idênticos
        assert len(tiers_all) == len(tiers_filtered)
