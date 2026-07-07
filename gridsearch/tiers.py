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

Também implementa os passos 4–10 do Algoritmo 1 (PSLA4ML completo) via
``generate_psla4ml()``, que retorna uma lista de objetos ``Tier``.

Exemplo de uso::

    from gridsearch.tiers import generate_psla4ml

    all_traces = state["results"]

    tiers = generate_psla4ml(
        results=all_traces,
        k=2,
        metrics=["train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"],
    )

    for tier in tiers:
        print(tier.hardware, tier.discretized)
        # GPU  {"train_time_sec": "< 5000", "cost_usd": "≥ 1.2", ...}

Autor: Gustavo Alexandre
"""

import logging
import statistics
from dataclasses import dataclass, field, asdict
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


# ============================================================================
# ENTIDADE TIER — Representa um nível de serviço do PSLA4ML
# ============================================================================

@dataclass
class Tier:
    """Representa um nível de serviço (tier) do PSLA4ML.

    Cada instância corresponde a uma configuração identificada pela consulta
    k-Skyband que oferece um compromisso distinto entre as métricas
    consideradas (cf. Seção 3.1 do artigo).

    Os atributos de configuração identificam o workflow (arquitetura + dados
    + hiperparâmetros livres), enquanto os campos ``_bin`` expressam os
    intervalos discretizados das métricas (Passo 9 do Algoritmo 1).

    Attributes:
        model: Arquitetura do modelo (componente ``A`` do TrainingTemplate).
        dataset: Identificador do dataset de treino (componente ``D``).
        learning_rate: Taxa de aprendizado usada no experimento.
        batch_size: Tamanho do batch.
        optimizer: Otimizador utilizado.
        dropout: Taxa de dropout (parte dos hiperparâmetros estruturais ``H``).
        hardware: Tipo de hardware (``"cpu"``, ``"gpu"`` ou ``"tpu"``).
        discretized: Intervalos discretizados das métricas, ex.:
            ``{"train_time_sec": "< 5000", "cost_usd": "≥ 1.2"}``.
        raw_metrics: Valores contínuos originais das métricas do trace.
        domination_count: Número de pontos que dominam este tier no espaço
            multicritério (0 = frente de Pareto).
        k: Ordem do Skyband usado para selecionar este tier.
        experiment_id: UUID do experimento de origem (quando disponível).
        grid_experiment_idx: Índice na grade de hiperparâmetros.
        selected_environment: Nome do ambiente de nuvem (ex.: ``"gcp"``).
    """

    # — Identificação do workflow —
    model: str = ""
    dataset: str = ""
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    optimizer: Optional[str] = None
    dropout: Optional[float] = None
    hardware: str = ""

    # — Métricas discretizadas (Passo 9) —
    discretized: Dict[str, str] = field(default_factory=dict)

    # — Valores brutos de referência —
    raw_metrics: Dict[str, float] = field(default_factory=dict)

    # — Metadados Skyband —
    domination_count: int = 0
    k: int = 1

    # — Rastreabilidade —
    experiment_id: Optional[str] = None
    grid_experiment_idx: Optional[int] = None
    selected_environment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializa o Tier para dicionário JSON-compatível."""
        return asdict(self)


# ============================================================================
# HELPERS DE EXTRAÇÃO — lêem campos dos dicts de resultado do grid search
# ============================================================================

def _extract_hardware(result: Dict[str, Any]) -> str:
    """Extrai o tipo de hardware do resultado.

    Hierarquia de busca:
    1. ``result["environment"]["device_type"]`` (produção, via ``build_result_dict``)
    2. ``result["selected_environment"]`` (nome do ambiente do multienv grid)
    3. ``result["grid_params"]["environment"]`` (chave adicionada pelo ``generate_parameter_grid``)
    4. String vazia como fallback.
    """
    env = result.get("environment", {})
    if isinstance(env, dict) and env.get("device_type"):
        return str(env["device_type"]).lower()
    se = result.get("selected_environment")
    if se:
        return str(se).lower()
    gp = result.get("grid_params", {})
    if isinstance(gp, dict) and gp.get("environment"):
        return str(gp["environment"]).lower()
    return ""


def _extract_hyperparam(result: Dict[str, Any], key: str, aliases: List[str]) -> Any:
    """Extrai um hiperparâmetro de um resultado, tentando múltiplos caminhos.

    Busca em:
    1. ``result["hyperparameters"][key]`` (resultado de produção)
    2. ``result["grid_params"][alias]`` para cada alias
    3. Raiz do dicionário com o próprio key e aliases
    """
    hp = result.get("hyperparameters", {})
    if isinstance(hp, dict):
        for k in [key] + aliases:
            if k in hp:
                return hp[k]
    gp = result.get("grid_params", {})
    if isinstance(gp, dict):
        for k in [key] + aliases:
            if k in gp:
                return gp[k]
    for k in [key] + aliases:
        if k in result:
            return result[k]
    return None


def _extract_raw_metrics(result: Dict[str, Any], metrics: List[str]) -> Dict[str, float]:
    """Extrai os valores brutos das métricas de um resultado."""
    return {
        m: v
        for m in metrics
        if (v := _extract_metric_value(result, m)) != float("inf")
    }


def _build_tier(
    result: Dict[str, Any],
    k: int,
    model: str,
    dataset: str,
    metrics: List[str],
) -> "Tier":
    """Constrói um Tier a partir de um resultado enriquecido com 'discretized'."""
    lr_raw = _extract_hyperparam(result, "learning_rate", ["lr"])
    bs_raw = _extract_hyperparam(result, "batch_size", ["bs", "batch"])
    opt_raw = _extract_hyperparam(result, "optimizer", ["opt"])
    do_raw = _extract_hyperparam(result, "dropout", ["do"])

    lr = float(lr_raw) if lr_raw is not None else None
    bs = int(bs_raw) if bs_raw is not None else None
    opt = str(opt_raw) if opt_raw is not None else None
    dropout = float(do_raw) if do_raw is not None else None

    exp_block = result.get("experiment", {})
    exp_id = exp_block.get("id") if isinstance(exp_block, dict) else None

    return Tier(
        model=model,
        dataset=dataset,
        learning_rate=lr,
        batch_size=bs,
        optimizer=opt,
        dropout=dropout,
        hardware=_extract_hardware(result),
        discretized=dict(result.get("discretized", {})),
        raw_metrics=_extract_raw_metrics(result, metrics),
        domination_count=int(result.get("domination_count", 0)),
        k=k,
        experiment_id=exp_id,
        grid_experiment_idx=result.get("grid_experiment_idx"),
        selected_environment=result.get("selected_environment"),
    )


# ============================================================================
# GENERATE_PSLA4ML — Algoritmo 1 completo (passos 4–10)
# ============================================================================

def generate_psla4ml(
    results: List[Dict[str, Any]],
    k: int = 1,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    strategy: str = "median",
    sla_constraints: Optional[Dict[str, float]] = None,
    model: str = "BERT-PLI",
    dataset: str = "COLLIE",
) -> List["Tier"]:
    """Gera os tiers do PSLA4ML executando o Algoritmo 1 do artigo.

    Implementa os passos 4–10 do Algoritmo 1 (Seção 3.2):

    .. code-block:: text

        4: S ← Skyband_k(P)
        5: para cada configuração p em S faça
        6:   Criar tier R
        7:   Adicionar R ao conjunto T
        8: fim para
        9: Discretizar métricas contínuas em T
       10: retorne T

    Os passos 1–3 (extração de traces do banco de proveniência e cálculo de
    métricas derivadas) são responsabilidade do chamador.

    Args:
        results: Conjunto de traces ``P`` — tipicamente todos os resultados
            do grid search carregados do arquivo de estado.
        k: Ordem do Skyband (``k=1`` equivale à frente de Pareto).
        metrics: Métricas usadas para dominância e discretização.
            Padrão: ``DEFAULT_METRICS`` (5 métricas de recurso).
        thresholds: Limiares explícitos ``{métrica: valor}`` para a
            discretização.  Quando ``None``, os limiares são calculados
            como a mediana do conjunto ``results`` (mesma estratégia
            usada no artigo).
        strategy: Estratégia de cálculo automático dos limiares:
            ``"median"`` (padrão), ``"mean"``, ``"q1"``, ``"q3"``.
        sla_constraints: Constraints de SLA aplicadas antes da dominância
            (ex.: ``{"cost_usd": 2.0}``).  ``None`` = sem restrição.
        model: Identificador da arquitetura do modelo (componente ``A``
            do TrainingTemplate, ex.: ``"BERT-PLI"``).
        dataset: Identificador do dataset de treino (componente ``D``,
            ex.: ``"COLLIE"``).

    Returns:
        Lista de :class:`Tier`, um por ponto no conjunto k-Skyband,
        ordenados por ``domination_count`` crescente (frente de Pareto
        primeiro).

    Example::

        tiers = generate_psla4ml(all_traces, k=2,
                                 metrics=["train_time_sec", "cost_usd"])
        for t in tiers:
            print(t.hardware, t.domination_count, t.discretized)
    """
    if metrics is None:
        metrics = DEFAULT_METRICS[:]

    # Passo 4: S ← Skyband_k(P)
    from .dominance import skyband_query
    skyband_results = skyband_query(
        results,
        k=k,
        sla_constraints=sla_constraints,
        metrics=metrics,
        minimize=[True] * len(metrics),
    )

    if not skyband_results:
        logger.warning("Skyband_k=%d retornou conjunto vazio; nenhum tier gerado.", k)
        return []

    # Passos 5–8: criar Tier R para cada p ∈ S após discretização (Passo 9)
    disc_results = discretize_metrics(
        results=skyband_results,
        metrics=metrics,
        thresholds=thresholds,
        strategy=strategy,
        reference_results=results,   # limiares calculados sobre TODO o conjunto
    )

    tiers: List[Tier] = [
        _build_tier(r, k=k, model=model, dataset=dataset, metrics=metrics)
        for r in disc_results
    ]

    # Ordena por domination_count (frente de Pareto primeiro)
    tiers.sort(key=lambda t: t.domination_count)

    logger.info(
        "PSLA4ML gerado: k=%d | %d tiers | modelo=%s | dataset=%s",
        k, len(tiers), model, dataset,
    )
    return tiers
