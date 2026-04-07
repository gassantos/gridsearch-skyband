"""
Geração de relatórios e exportação de análises
================================================

Gera relatórios textuais e exporta análises em JSON.
Inclui a interface CLI standalone para análise de resultados.

Autor: Gustavo Alexandre
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from .statistics import compute_descriptive_statistics
from .correlations import analyze_correlations
from .ranking import rank_configurations
from .hyperparameters import find_best_value_per_hyperparameter

logger = logging.getLogger(__name__)


def export_analysis_to_json(analysis_data: Dict[str, Any], output_file: Path):
    """
    Exporta análise completa para JSON.

    Args:
        analysis_data: Dados da análise
        output_file: Arquivo de saída
    """
    logger.info(f"Exportando análise para {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2, default=str)

    logger.info("Análise exportada com sucesso")


def generate_analysis_report(results: List[Dict[str, Any]]) -> str:
    """
    Gera relatório detalhado de análise.

    Args:
        results: Lista de resultados

    Returns:
        String formatada com relatório completo
    """
    report = []
    report.append("=" * 80)
    report.append("ANÁLISE DETALHADA - GRID SEARCH")
    report.append("=" * 80)
    report.append("")

    # Estatísticas descritivas
    stats = compute_descriptive_statistics(results)

    report.append("ESTATÍSTICAS DESCRITIVAS:")
    report.append("")

    if "train_time" in stats:
        report.append("  Tempo de Treinamento (segundos):")
        t = stats["train_time"]
        if t["count"] > 0:
            report.append(f"    Média: {t['mean']:.2f}")
            report.append(f"    Mediana: {t['median']:.2f}")
            report.append(f"    Desvio Padrão: {t['stdev']:.2f}")
            report.append(f"    Mínimo: {t['min']:.2f}")
            report.append(f"    Máximo: {t['max']:.2f}")
        report.append("")

    if "energy_kwh" in stats:
        report.append("  Energia Consumida (kWh):")
        e = stats["energy_kwh"]
        if e["count"] > 0:
            report.append(f"    Média: {e['mean']:.4f}")
            report.append(f"    Mediana: {e['median']:.4f}")
            report.append(f"    Desvio Padrão: {e['stdev']:.4f}")
            report.append(f"    Mínimo: {e['min']:.4f}")
            report.append(f"    Máximo: {e['max']:.4f}")
        report.append("")

    if "peak_ram_mb" in stats:
        report.append("  Uso de RAM Pico (MB):")
        r = stats["peak_ram_mb"]
        if r["count"] > 0:
            report.append(f"    Média: {r['mean']:.2f}")
            report.append(f"    Mediana: {r['median']:.2f}")
            report.append(f"    Desvio Padrão: {r['stdev']:.2f}")
            report.append(f"    Mínimo: {r['min']:.2f}")
            report.append(f"    Máximo: {r['max']:.2f}")
        report.append("")

    # Correlações
    correlations = analyze_correlations(results)

    if correlations:
        report.append("CORRELAÇÕES:")
        report.append("")
        for corr_name, corr_value in correlations.items():
            if corr_value is not None:
                report.append(f"  {corr_name}: {corr_value:.3f}")
        report.append("")

    # Ranking top 10
    ranked = rank_configurations(results)

    if ranked:
        report.append("TOP 10 CONFIGURAÇÕES:")
        report.append("")

        for i, config in enumerate(ranked[:10], 1):
            report.append(
                f"  {i}. Experimento {config['experiment_idx']} (Score: {config['score']:.3f})"
            )
            report.append(f"     Parâmetros: {config['params']}")
            report.append("")

    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Executa análise standalone dos resultados."""
    import argparse

    from . import GRID_RESULTS_FILE, ANALYSIS_DIR

    parser = argparse.ArgumentParser(description="Análise de resultados do Grid Search")

    parser.add_argument(
        "--results-file",
        type=str,
        default=str(GRID_RESULTS_FILE),
        help="Caminho do arquivo de resultados JSON",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ANALYSIS_DIR),
        help="Diretório de saída para análises",
    )

    args = parser.parse_args()

    results_file = Path(args.results_file)

    if not results_file.exists():
        logger.error(f"Arquivo de resultados não encontrado: {results_file}")
        sys.exit(1)

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    logger.info(f"Carregados {len(results)} resultados")

    stats = compute_descriptive_statistics(results)
    correlations = analyze_correlations(results)
    ranked = rank_configurations(results)
    best_per_param = find_best_value_per_hyperparameter(results)

    full_analysis = {
        "descriptive_statistics": stats,
        "correlations": correlations,
        "ranking": ranked,
        "best_values_per_hyperparameter": best_per_param,
    }

    output_dir = Path(args.output_dir)
    export_analysis_to_json(full_analysis, output_dir / "full_analysis.json")

    report = generate_analysis_report(results)

    print("\n" + report)

    report_file = output_dir / "analysis_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, "w") as f:
        f.write(report)

    logger.info(f"Relatório salvo em: {report_file}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    main()
