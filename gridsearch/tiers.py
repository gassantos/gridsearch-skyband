"""
PSLA4ML Tier Generation — Discretização de Métricas (Passo 9 do Algoritmo 1)
==============================================================================

Implementa a etapa de discretização das métricas contínuas do conjunto
k-Skyband em intervalos baseados em limiares, conforme descrito na
Seção 3.2 do artigo PSLA4ML:

    "Para cada trace retornado pela consulta k-Skyband, realiza-se a
     discretização das métricas contínuas, de modo que os tiers sejam
     associados a intervalos, e não a valores exatos."

Cada ponto retornado por ``skyband_query`` tem suas métricas convertidas
em intervalos do tipo ``"< 5000"`` ou ``"≥ 5000"`` (cf. Tabela 3 do artigo).

Exemplo de uso::

    from gridsearch.tiers import discretize_metrics

    # Todos os traces do grid search
    all_traces = state["results"]

    # Skyband de ordem k=2
    from gridsearch.dominance import skyband_query
    sb = skyband_query(all_traces, k=2)

    # Discretiza as métricas dos pontos do Skyband usando a mediana
    # do conjunto completo como limiar
    tiers = discretize_metrics(
        results=sb,
        metrics=["train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"],
        reference_results=all_traces,
        strategy="median",
    )

    for tier in tiers:
        print(tier["discretized"])
        # {"train_time_sec": "< 5000", "energy_kwh": "< 0.2", ...}

Autor: Gustavo Alexandre
"""

import logging
import statistics
from typing import Any, Dict, List, Optional

from .dominance import DEFAULT_METRICS, _extract_metric_value

logger = logging.getLogger(__name__)

# Estratégias disponíveis para cálculo automático de limiares
DISCRETIZATION_STRATEGIES = ("median", "mean", "q1", "q3")


def compute_thresholds(
    results: List[Dict[str, Any]],
    metrics: List[str],
    strategy: str = "median",
    explicit_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Calcula os limiares de discretização para cada métrica.

    Os limiares são calculados a partir dos valores observados nos
    resultados fornecidos (tipicamente o conjunto completo de traces,
    não apenas o Skyband).  Limiares explícitos em ``explicit_thresholds``
    sobrescrevem o cálculo automático por métrica.

    Args:
        results: Lista de resultados de experimentos usados como referência.
        metrics: Lista de nomes de métricas a calcular limiares.
        strategy: Estratégia de cálculo automático:

            - ``"median"`` — mediana dos valores (padrão; usada no artigo)
            - ``"mean"``   — média aritmética
            - ``"q1"``     — primeiro quartil (25%)
            - ``"q3"``     — terceiro quartil (75%)

        explicit_thresholds: Dicionário ``{métrica: limiar}`` com limiares
            explícitos.  Sobrescreve os calculados automaticamente para
            as métricas informadas.

    Returns:
        Dicionário ``{métrica: limiar}`` com todos os limiares calculados.

    Raises:
        ValueError: Quando ``strategy`` não é uma das opções suportadas.
    """
    if strategy not in DISCRETIZATION_STRATEGIES:
        raise ValueError(
            f"Estratégia inválida: '{strategy}'. "
            f"Disponíveis: {DISCRETIZATION_STRATEGIES}"
        )

    valid_results = [r for r in results if r.get("status") == "success"]
    thresholds: Dict[str, float] = {}

    for metric in metrics:
        values = [
            _extract_metric_value(r, metric)
            for r in valid_results
        ]
        values = [v for v in values if v != float("inf")]

        if not values:
            logger.warning(
                "Nenhum valor válido para métrica '%s'; limiar não calculado.",
                metric,
            )
            continue

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        if strategy == "median":
            thresholds[metric] = statistics.median(sorted_vals)
        elif strategy == "mean":
            thresholds[metric] = statistics.mean(sorted_vals)
        elif strategy == "q1":
            thresholds[metric] = sorted_vals[max(0, (n - 1) // 4)]
        elif strategy == "q3":
            thresholds[metric] = sorted_vals[min(n - 1, (3 * (n - 1)) // 4)]

        logger.debug(
            "Limiar '%s' (%s de %d valores): %.6g",
            metric, strategy, n, thresholds[metric],
        )

    if explicit_thresholds:
        for m, v in explicit_thresholds.items():
            if m in metrics:
                thresholds[m] = float(v)
                logger.debug("Limiar explícito '%s': %.6g", m, v)

    return thresholds


def _format_threshold(threshold: float) -> str:
    """Formata o valor do limiar para exibição compacta nos intervalos.

    Regras (para reproduzir a notação da Tabela 3 do artigo):
    - Valores ≥ 1000 sem parte decimal → inteiro
    - Valores < 0.001 → notação científica
    - Demais → máximo 4 algarismos significativos
    """
    if threshold == 0.0:
        return "0"
    abs_t = abs(threshold)
    if abs_t >= 1000:
        return str(int(threshold)) if threshold == int(threshold) else f"{threshold:.4g}"
    if abs_t < 0.001:
        return f"{threshold:.3e}"
    if threshold == int(threshold):
        return str(int(threshold))
    return f"{threshold:.4g}"


def _interval_for(value: float, threshold: float) -> str:
    """Retorna a string de intervalo para um valor dado o limiar."""
    thr_str = _format_threshold(threshold)
    return f"< {thr_str}" if value < threshold else f"\u2265 {thr_str}"


def discretize_metrics(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    strategy: str = "median",
    reference_results: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Discretiza métricas contínuas em intervalos baseados em limiares.

    Implementa o **Passo 9 do Algoritmo 1** do artigo PSLA4ML::

        9: Discretizar métricas contínuas em T

    Para cada resultado, adiciona a chave ``"discretized"`` com um
    dicionário mapeando cada métrica ao intervalo correspondente::

        "discretized": {
            "train_time_sec":   "< 5000",
            "energy_kwh":       "< 0.2",
            "emissions_kg_co2": "< 0.05",
            "cost_usd":         "≥ 1.2"
        }

    Os limiares são calculados a partir de ``reference_results`` (ou dos
    próprios ``results`` quando não fornecido).  Limiares explícitos em
    ``thresholds`` sobrescrevem o cálculo automático por métrica.

    O conjunto ``reference_results`` deve ser o conjunto **completo** de
    traces (não apenas o Skyband), de modo que os limiares reflitam a
    distribuição real dos dados (cf. Figura 2 do artigo).

    Args:
        results: Resultados a discretizar (tipicamente o conjunto k-Skyband).
            Não é modificado in-place — retorna cópias rasas.
        metrics: Métricas a discretizar.  Padrão: ``DEFAULT_METRICS``.
        thresholds: Limiares explícitos ``{métrica: valor}``.  Quando
            fornecido para uma métrica, sobrescreve o cálculo automático.
        strategy: Estratégia de cálculo automático dos limiares:
            ``"median"`` (padrão), ``"mean"``, ``"q1"``, ``"q3"``.
        reference_results: Conjunto de referência para o cálculo dos
            limiares.  Recomendado: todos os traces do grid search.
            Quando ``None``, usa ``results``.

    Returns:
        Lista de cópias dos resultados com a chave ``"discretized"``
        adicionada a cada elemento.  Resultados sem nenhum limiar
        calculável retornam ``"n/a"`` para as métricas afetadas.

    Example::

        all_traces = state["results"]
        sb = skyband_query(all_traces, k=2)

        tiers = discretize_metrics(
            results=sb,
            metrics=["train_time_sec", "cost_usd"],
            reference_results=all_traces,
        )
        # tiers[0]["discretized"] == {"train_time_sec": "< 5000", "cost_usd": "≥ 1.2"}
    """
    if metrics is None:
        metrics = DEFAULT_METRICS[:]

    ref = reference_results if reference_results is not None else results

    computed = compute_thresholds(
        results=ref,
        metrics=metrics,
        strategy=strategy,
        explicit_thresholds=thresholds,
    )

    if not computed:
        logger.warning(
            "Nenhum limiar calculado para %s; campos 'discretized' preenchidos com 'n/a'.",
            metrics,
        )
        return [
            {**dict(r), "discretized": {m: "n/a" for m in metrics},
             "discretization_thresholds": {}}
            for r in results
        ]

    discretized: List[Dict[str, Any]] = []
    for result in results:
        r = dict(result)
        disc: Dict[str, str] = {}
        for metric in metrics:
            if metric not in computed:
                continue
            value = _extract_metric_value(r, metric)
            disc[metric] = "n/a" if value == float("inf") else _interval_for(
                value, computed[metric]
            )
        r["discretized"] = disc
        r["discretization_thresholds"] = dict(computed)
        discretized.append(r)

    logger.info(
        "Discretização concluída: %d resultados | %d métricas | limiares=%s",
        len(discretized),
        len(computed),
        {m: _format_threshold(v) for m, v in computed.items()},
    )
    return discretized
