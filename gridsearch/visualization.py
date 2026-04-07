"""
Skyband Visualization — Gráficos e relatórios textuais
======================================================

Visualização 2D da frente de Pareto e geração de relatórios
textuais do Skyband.

Autor: Gustavo Alexandre
Data: 2026-03-01
"""

import logging
from typing import Any, Dict, List, Optional

from .dominance import (
    DEFAULT_METRICS,
    _extract_metric_value,
    skyband_query,
    sla_filter,
)

logger = logging.getLogger(__name__)


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
