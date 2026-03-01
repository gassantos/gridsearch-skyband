"""
Skyband Query Engine - BERT-PLI
================================

Motor de consulta Skyband para seleção multicriterio de configurações
de hiperparâmetros com suporte a personalização de SLA.

Implementa dominância de Pareto estendida com filtro de SLA, permitindo
que o usuário defina constraints de acordo com seu ambiente computacional
(local, GCP, AWS, Azure, Colab) e obtenha o conjunto de configurações
não-dominadas que respeitam seus requisitos.

Conceitos:
    - Dominância de Pareto: e_i domina e_j se e_i é melhor ou igual em
      todos os critérios e estritamente melhor em pelo menos um.
    - Skyline (k=1): conjunto de pontos não-dominados por nenhum outro.
    - Skyband(k): conjunto de pontos dominados por menos de k outros pontos.
    - SLA Filter: restrições de contexto aplicadas antes da dominância.

Uso:
    from gridsearch.skyband import skyband_query, sla_filter, pareto_front

    # Skyline puro (k=1) sem SLA
    front = pareto_front(results)

    # Skyband de ordem 3 com SLA de custo e tempo
    recs = skyband_query(
        results,
        k=3,
        sla_constraints={"cost_usd": 5.0, "train_time_sec": 7200},
        metrics=["train_time_sec", "cost_usd", "energy_kwh"],
    )

    # Comparar Skyband vs ranking escalar atual
    report = compare_skyband_vs_ranking(results, sla={"cost_usd": 5.0})

Autor: Gustavo Alexandre
Data: 2026-03-01
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Métricas padrão (todas a minimizar) alinhadas com os campos de result["resources"]
DEFAULT_METRICS: List[str] = [
    "train_time_sec",
    "energy_kwh",
    "total_gflops",
    "emissions_kg_co2",
    "cost_usd",
]

# Por padrão, todas as métricas são minimizadas
DEFAULT_MINIMIZE: List[bool] = [True] * len(DEFAULT_METRICS)


# ============================================================================
# UTILITÁRIOS INTERNOS
# ============================================================================

def _extract_metric_value(result: Dict[str, Any], metric: str) -> float:
    """
    Extrai o valor de uma métrica de um resultado de experimento.

    Procura primeiro em result["resources"] (métricas de recursos) e,
    caso não encontre, busca diretamente na raiz do dicionário.

    Args:
        result: Dicionário de resultado de um experimento.
        metric: Nome da métrica a extrair.

    Returns:
        Valor numérico da métrica ou float('inf') se ausente/None.
    """
    resources = result.get("resources", {})
    value = resources.get(metric)

    if value is None:
        value = result.get(metric)

    if value is None:
        return float("inf")

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _build_vector(
    result: Dict[str, Any],
    metrics: List[str],
    minimize: List[bool],
) -> List[float]:
    """
    Constrói o vetor de comparação para cálculo de dominância.

    Valores são normalizados pelo sinal de minimização para que
    dominância sempre signifique "menor é melhor".

    Args:
        result:   Dicionário de resultado.
        metrics:  Lista de nomes de métricas.
        minimize: Lista de booleanos (True = minimizar, False = maximizar).

    Returns:
        Lista de floats representando o ponto no espaço de critérios.
    """
    return [
        (1.0 if mini else -1.0) * _extract_metric_value(result, m)
        for m, mini in zip(metrics, minimize)
    ]


# ============================================================================
# FILTRO DE SLA
# ============================================================================

def sla_filter(
    results: List[Dict[str, Any]],
    constraints: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Filtra experimentos que violam as constraints de SLA informadas.

    Constraints são aplicadas como limites superiores sobre as métricas
    de recursos. Apenas experimentos que satisfazem **todas** as constraints
    são mantidos.

    Args:
        results:     Lista de resultados de experimentos.
        constraints: Dicionário {nome_metrica: valor_maximo}.
                     Exemplos:
                       {"train_time_sec": 7200}        → máx 2h de treino
                       {"cost_usd": 5.0}               → máx U$ 5,00
                       {"emissions_kg_co2": 0.01}      → máx 10g CO₂
                       {"total_gflops": 500.0}         → máx 500 GFLOPS por epoch

    Returns:
        Subconjunto de results que satisfaz todas as constraints.

    Exemplo:
        >>> admissible = sla_filter(results, {"cost_usd": 5.0, "train_time_sec": 3600})
        >>> len(admissible)
        42
    """
    admissible = []
    rejected = 0

    for r in results:
        if r.get("status") != "success":
            rejected += 1
            continue

        if not constraints:
            admissible.append(r)
            continue

        satisfies_all = all(
            _extract_metric_value(r, metric) <= threshold
            for metric, threshold in constraints.items()
        )

        if satisfies_all:
            admissible.append(r)
        else:
            rejected += 1

    logger.info(
        "SLA filter: %d admissíveis | %d rejeitados | constraints=%s",
        len(admissible), rejected, constraints,
    )
    return admissible


# ============================================================================
# DOMINÂNCIA DE PARETO
# ============================================================================

def dominates(
    vec_i: List[float],
    vec_j: List[float],
) -> bool:
    """
    Verifica se o ponto vec_i domina o ponto vec_j (Pareto).

    e_i domina e_j se:
        - e_i é melhor ou igual em TODOS os critérios, E
        - e_i é estritamente melhor em PELO MENOS UM critério.

    Args:
        vec_i: Vetor de métricas do experimento i (já normalizado por sinal).
        vec_j: Vetor de métricas do experimento j (já normalizado por sinal).

    Returns:
        True se vec_i domina vec_j, False caso contrário.
    """
    at_least_one_better = False

    for a, b in zip(vec_i, vec_j):
        if a > b:
            return False          # pior em algum critério → não domina
        if a < b:
            at_least_one_better = True

    return at_least_one_better


# ============================================================================
# FRENTE DE PARETO (k=1)
# ============================================================================

def pareto_front(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    minimize: Optional[List[bool]] = None,
    sla_constraints: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Retorna a frente de Pareto pura (Skyband de ordem k=1).

    Equivalente a skyband_query(results, k=1, ...).

    Args:
        results:         Lista de resultados de experimentos.
        metrics:         Critérios usados na dominância.
                         Padrão: ["train_time_sec", "energy_kwh",
                                  "total_gflops", "emissions_kg_co2", "cost_usd"]
        minimize:        Lista de booleanos correspondente a metrics.
                         True = minimizar (padrão para todos).
        sla_constraints: Constraints de SLA aplicadas antes da dominância.

    Returns:
        Lista de experimentos na frente de Pareto, ordenados por índice
        original, enriquecidos com a chave "domination_count" = 0.

    Exemplo:
        >>> front = pareto_front(results, metrics=["train_time_sec", "cost_usd"])
        >>> for r in front:
        ...     print(r["grid_experiment_idx"], r["domination_count"])
    """
    return skyband_query(
        results,
        k=1,
        sla_constraints=sla_constraints,
        metrics=metrics,
        minimize=minimize,
    )


# ============================================================================
# CONSULTA SKYBAND (k ≥ 1)
# ============================================================================

def skyband_query(
    results: List[Dict[str, Any]],
    k: int = 1,
    sla_constraints: Optional[Dict[str, float]] = None,
    metrics: Optional[List[str]] = None,
    minimize: Optional[List[bool]] = None,
) -> List[Dict[str, Any]]:
    """
    Consulta Skyband de ordem k com filtro de SLA personalizado.

    Retorna os experimentos dominados por **menos de k** outros pontos,
    aplicando primeiro o filtro de SLA sobre o espaço de candidatos.

    Matematicamente:
        Skyband_k(E_SLA) = { e ∈ E_SLA | |{ e' ∈ E_SLA : e' ≺ e }| < k }

    Onde:
        E_SLA = sla_filter(E, constraints)
        k=1   → frente de Pareto pura (nenhum ponto domina e)
        k=2   → inclui pontos dominados por exatamente 1 outro ponto
        k=n   → inclui praticamente todos os pontos admissíveis

    Args:
        results:         Lista de resultados de experimentos.
        k:               Ordem do Skyband (>= 1). Maior k = conjunto maior.
        sla_constraints: Constraints de SLA {metrica: valor_maximo}.
                         None = sem filtro de SLA.
        metrics:         Lista de nomes de métricas para dominância.
                         Padrão: DEFAULT_METRICS (5 critérios).
        minimize:        Lista de booleanos por métrica. True = minimizar.
                         Padrão: todos True.

    Returns:
        Lista de dicionários de experimento, cada um com a chave extra:
          - "domination_count": int — quantos outros experimentos o dominam
          - "skyband_rank": int — posição no Skyband (0 = Pareto front)
        Ordenado por domination_count (crescente) e depois por índice.

    Raises:
        ValueError: Se k < 1 ou se metrics e minimize têm tamanhos diferentes.

    Exemplo:
        >>> recs = skyband_query(
        ...     results,
        ...     k=3,
        ...     sla_constraints={"cost_usd": 10.0, "train_time_sec": 7200},
        ...     metrics=["train_time_sec", "cost_usd"],
        ... )
        >>> for r in recs:
        ...     idx = r["grid_experiment_idx"]
        ...     dom = r["domination_count"]
        ...     params = r["grid_params"]
        ...     print(f"Exp {idx:03d} | dominado por {dom} | {params}")
    """
    if k < 1:
        raise ValueError(f"k deve ser >= 1, recebido: {k}")

    if metrics is None:
        metrics = DEFAULT_METRICS[:]
    if minimize is None:
        minimize = [True] * len(metrics)

    if len(metrics) != len(minimize):
        raise ValueError(
            f"metrics ({len(metrics)}) e minimize ({len(minimize)}) "
            "devem ter o mesmo tamanho."
        )

    # 1. Aplica filtro de SLA
    candidates = sla_filter(results, sla_constraints or {})

    if not candidates:
        logger.warning(
            "Nenhum experimento admissível após filtro SLA (constraints=%s). "
            "Verifique os thresholds ou execute mais experimentos.",
            sla_constraints,
        )
        return []

    # 2. Constrói vetores de métricas normalizados por sinal
    vecs = [_build_vector(r, metrics, minimize) for r in candidates]

    # 3. Conta quantos pontos dominam cada candidato
    n = len(candidates)
    domination_count = [0] * n

    for i in range(n):
        for j in range(n):
            if i != j and dominates(vecs[j], vecs[i]):
                domination_count[i] += 1

    # 4. Seleciona o Skyband_k: dominados por < k outros pontos
    skyband: List[Dict[str, Any]] = []

    for i, result in enumerate(candidates):
        if domination_count[i] < k:
            enriched = dict(result)
            enriched["domination_count"] = domination_count[i]
            skyband.append(enriched)

    # 5. Ordena por (domination_count, experiment_idx)
    skyband.sort(key=lambda r: (r["domination_count"], r.get("grid_experiment_idx", 0)))

    # 6. Atribui skyband_rank (0 = frente de Pareto)
    for rank, r in enumerate(skyband):
        r["skyband_rank"] = rank

    logger.info(
        "Skyband_%d: %d/%d candidatos selecionados | metrics=%s | sla=%s",
        k, len(skyband), len(candidates), metrics, sla_constraints,
    )

    return skyband


# ============================================================================
# COMPARAÇÃO SKYBAND vs RANKING ESCALAR
# ============================================================================

def compare_skyband_vs_ranking(
    results: List[Dict[str, Any]],
    sla: Optional[Dict[str, float]] = None,
    metrics: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    k: int = 1,
) -> Dict[str, Any]:
    """
    Compara o resultado da consulta Skyband com o ranking escalar ponderado.

    O ranking escalar é o método atual em analysis.rank_configurations(),
    que colapsa todos os critérios em um único score via normalização
    min-max ponderada. Esta função evidencia as diferenças qualitativas
    entre as duas abordagens.

    Args:
        results:  Lista de resultados de experimentos.
        sla:      Constraints de SLA aplicadas a ambas as abordagens.
        metrics:  Critérios usados (padrão: DEFAULT_METRICS).
        weights:  Pesos para o ranking escalar (padrão: iguais).
        k:        Ordem do Skyband.

    Returns:
        Dicionário com:
          - "skyband":             lista de experimentos no Skyband_k
          - "scalar_top":          lista dos top-|Skyband| pelo ranking escalar
          - "only_in_skyband":     índices presentes no Skyband mas não no top escalar
          - "only_in_scalar":      índices presentes no top escalar mas não no Skyband
          - "intersection":        índices em ambos
          - "jaccard_similarity":  |interseção| / |união|
          - "skyband_size":        tamanho do Skyband
          - "scalar_top_size":     tamanho do conjunto escalar comparado
          - "k":                   ordem do Skyband utilizada
          - "sla":                 constraints de SLA aplicadas
          - "metrics":             métricas utilizadas

    Exemplo:
        >>> report = compare_skyband_vs_ranking(
        ...     results,
        ...     sla={"cost_usd": 5.0},
        ...     metrics=["train_time_sec", "cost_usd"],
        ...     k=3,
        ... )
        >>> print(f"Jaccard: {report['jaccard_similarity']:.2f}")
        >>> print(f"Somente no Skyband: {report['only_in_skyband']}")
    """
    if metrics is None:
        metrics = DEFAULT_METRICS[:]
    if weights is None:
        weights = [1.0] * len(metrics)

    # --- Skyband ---
    sb_results = skyband_query(results, k=k, sla_constraints=sla, metrics=metrics)
    sb_size = len(sb_results)
    sb_indices = {r.get("grid_experiment_idx") for r in sb_results}

    # --- Ranking escalar (reimplementado localmente para independência) ---
    candidates = sla_filter(results, sla or {})

    # Coleta valores brutos por métrica
    raw_values: List[List[Optional[float]]] = []
    for m in metrics:
        col = [_extract_metric_value(r, m) for r in candidates]
        raw_values.append(col)

    # Normalização min-max por coluna
    normalized: List[List[float]] = []
    for col in raw_values:
        valid = [v for v in col if v != float("inf")]
        if not valid:
            normalized.append([0.0] * len(col))
            continue
        mn, mx = min(valid), max(valid)
        rng = mx - mn if mx != mn else 1.0
        normalized.append(
            [(v - mn) / rng if v != float("inf") else 1.0 for v in col]
        )

    # Score escalar ponderado
    scored = []
    for i, r in enumerate(candidates):
        score = sum(weights[j] * normalized[j][i] for j in range(len(metrics)))
        scored.append((score, r.get("grid_experiment_idx"), r))

    scored.sort(key=lambda x: x[0])
    scalar_top_n = scored[:sb_size]
    scalar_indices = {item[1] for item in scalar_top_n}

    # --- Análise de diferenças ---
    only_sb = sorted(sb_indices - scalar_indices)
    only_sc = sorted(scalar_indices - sb_indices)
    intersection = sorted(sb_indices & scalar_indices)
    union = sb_indices | scalar_indices
    jaccard = len(intersection) / len(union) if union else 1.0

    report = {
        "skyband": sb_results,
        "scalar_top": [item[2] for item in scalar_top_n],
        "only_in_skyband": only_sb,
        "only_in_scalar": only_sc,
        "intersection": intersection,
        "jaccard_similarity": jaccard,
        "skyband_size": sb_size,
        "scalar_top_size": sb_size,
        "k": k,
        "sla": sla,
        "metrics": metrics,
    }

    logger.info(
        "Skyband vs Escalar | k=%d | Jaccard=%.3f | "
        "só Skyband=%d | só Escalar=%d | interseção=%d",
        k, jaccard, len(only_sb), len(only_sc), len(intersection),
    )

    return report


# ============================================================================
# VISUALIZAÇÃO (opcional — requer matplotlib)
# ============================================================================

def plot_pareto_2d(
    results: List[Dict[str, Any]],
    x_metric: str = "train_time_sec",
    y_metric: str = "cost_usd",
    sla_constraints: Optional[Dict[str, float]] = None,
    k: int = 1,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plota a frente de Pareto 2D destacando o Skyband_k.

    Requer matplotlib. Se não estiver instalado, registra um aviso e retorna.

    Codificação visual:
        ★  Ponto na frente de Pareto (domination_count = 0)
        ●  Ponto no Skyband mas fora da frente de Pareto
        ·  Ponto fora do Skyband (dominado por >= k outros)
        ×  Ponto rejeitado pelo filtro de SLA

    Args:
        results:         Lista de resultados de experimentos.
        x_metric:        Métrica para o eixo X.
        y_metric:        Métrica para o eixo Y.
        sla_constraints: Constraints de SLA aplicadas.
        k:               Ordem do Skyband a destacar.
        title:           Título do gráfico. Padrão gerado automaticamente.
        save_path:       Caminho para salvar a imagem (PNG/PDF).
                         None = exibe interativamente.

    Returns:
        None
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning(
            "matplotlib não encontrado. Instale com: pip install matplotlib\n"
            "Visualização ignorada."
        )
        return

    successful = [r for r in results if r.get("status") == "success"]
    admitted = sla_filter(successful, sla_constraints or {})
    admitted_ids = {r.get("grid_experiment_idx") for r in admitted}

    # Skyband para destacar
    sb = skyband_query(
        results,
        k=k,
        sla_constraints=sla_constraints,
        metrics=[x_metric, y_metric],
        minimize=[True, True],
    )
    pareto_ids = {r.get("grid_experiment_idx") for r in sb if r["domination_count"] == 0}
    other_sb_ids = {r.get("grid_experiment_idx") for r in sb if r["domination_count"] > 0}

    fig, ax = plt.subplots(figsize=(9, 6))

    # Pontos rejeitados pelo SLA
    for r in successful:
        idx = r.get("grid_experiment_idx")
        if idx not in admitted_ids:
            ax.scatter(
                _extract_metric_value(r, x_metric),
                _extract_metric_value(r, y_metric),
                marker="x", color="lightgray", alpha=0.5, zorder=1,
            )

    # Pontos admissíveis fora do Skyband
    for r in admitted:
        idx = r.get("grid_experiment_idx")
        if idx not in pareto_ids and idx not in other_sb_ids:
            ax.scatter(
                _extract_metric_value(r, x_metric),
                _extract_metric_value(r, y_metric),
                marker=".", color="steelblue", alpha=0.4, zorder=2,
            )

    # Skyband fora da frente de Pareto
    for r in sb:
        if r["domination_count"] > 0:
            ax.scatter(
                _extract_metric_value(r, x_metric),
                _extract_metric_value(r, y_metric),
                marker="o", color="orange", s=60, alpha=0.8, zorder=3,
            )

    # Frente de Pareto
    pareto_x, pareto_y = [], []
    for r in sb:
        if r["domination_count"] == 0:
            pareto_x.append(_extract_metric_value(r, x_metric))
            pareto_y.append(_extract_metric_value(r, y_metric))
            ax.scatter(
                _extract_metric_value(r, x_metric),
                _extract_metric_value(r, y_metric),
                marker="*", color="crimson", s=120, zorder=4,
            )

    # Linha da frente de Pareto (step plot)
    if pareto_x:
        paired = sorted(zip(pareto_x, pareto_y))
        px, py = zip(*paired)
        ax.step(px, py, where="post", color="crimson", linewidth=1.2,
                linestyle="--", alpha=0.6, zorder=3)

    legend_handles = [
        mpatches.Patch(color="lightgray",  label="Rejeitado (viola SLA)"),
        mpatches.Patch(color="steelblue",  label="Admissível (fora Skyband)"),
        mpatches.Patch(color="orange",     label=f"Skyband_{k} (fora Pareto)"),
        mpatches.Patch(color="crimson",    label="Frente de Pareto (k=1)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    ax.set_xlabel(x_metric, fontsize=11)
    ax.set_ylabel(y_metric, fontsize=11)

    if title is None:
        sla_str = str(sla_constraints) if sla_constraints else "sem SLA"
        title = f"Frente de Pareto — Skyband_{k} | SLA: {sla_str}"
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Gráfico salvo em: %s", save_path)
    else:
        plt.show()

    plt.close(fig)


# ============================================================================
# INTERFACE AUXILIAR: RESUMO EM TEXTO
# ============================================================================

def skyband_report(
    results: List[Dict[str, Any]],
    k: int = 1,
    sla_constraints: Optional[Dict[str, float]] = None,
    metrics: Optional[List[str]] = None,
    minimize: Optional[List[bool]] = None,
) -> str:
    """
    Gera relatório textual do Skyband_k para um dado conjunto de resultados.

    Args:
        results:         Lista de resultados de experimentos.
        k:               Ordem do Skyband.
        sla_constraints: Constraints de SLA.
        metrics:         Métricas utilizadas na dominância.
        minimize:        Direção de otimização por métrica.

    Returns:
        String formatada com relatório completo do Skyband.

    Exemplo:
        >>> print(skyband_report(results, k=3, sla_constraints={"cost_usd": 5.0}))
    """
    if metrics is None:
        metrics = DEFAULT_METRICS[:]
    if minimize is None:
        minimize = [True] * len(metrics)

    sb = skyband_query(results, k=k, sla_constraints=sla_constraints,
                       metrics=metrics, minimize=minimize)

    lines = [
        "=" * 72,
        f"RELATÓRIO SKYBAND (k={k})",
        "=" * 72,
        f"  SLA constraints : {sla_constraints or 'nenhuma'}",
        f"  Métricas        : {metrics}",
        f"  Total de candidatos admissíveis : {len(sla_filter(results, sla_constraints or {}))}",
        f"  Tamanho do Skyband_{k}          : {len(sb)}",
        "",
        f"{'Rank':<5} {'Exp':>5} {'Dom':>4}  {'Parâmetros':<45}  Métricas",
        "-" * 72,
    ]

    for r in sb:
        idx = r.get("grid_experiment_idx", "?")
        dom = r.get("domination_count", "?")
        params = r.get("grid_params", {})
        rank = r.get("skyband_rank", "?")

        metric_vals = "  ".join(
            f"{m}={_extract_metric_value(r, m):.4g}" for m in metrics
        )

        params_str = str(params)[:44]
        lines.append(f"{rank:<5} {idx:>5} {dom:>4}  {params_str:<45}  {metric_vals}")

    lines.append("=" * 72)
    return "\n".join(lines)
