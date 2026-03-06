"""Testes de agregacao de backend no gridsearch.core."""

from gridsearch.core import analyze_results, generate_summary_report


def _sample_results():
    return [
        {
            "grid_experiment_idx": 0,
            "grid_params": {"learning_rate": 1e-5},
            "status": "success",
            "environment": {
                "backend_requested": "torch",
                "backend_resolved": "torch",
            },
            "resources": {"train_time_sec": 10.0, "energy_kwh": 0.1, "peak_ram_mb": 100.0},
        },
        {
            "grid_experiment_idx": 1,
            "grid_params": {"learning_rate": 2e-5},
            "status": "success",
            "environment": {
                "backend_requested": "jax",
                "backend_resolved": "torch",
            },
            "resources": {"train_time_sec": 11.0, "energy_kwh": 0.2, "peak_ram_mb": 120.0},
        },
        {
            "grid_experiment_idx": 2,
            "grid_params": {"learning_rate": 3e-5},
            "status": "failed",
            "error": "boom",
        },
    ]


def test_analyze_results_has_backend_breakdown():
    analysis = analyze_results(_sample_results())
    assert "backend_breakdown" in analysis
    assert analysis["backend_breakdown"]["torch"]["torch"] == 1
    assert analysis["backend_breakdown"]["jax"]["torch"] == 1


def test_report_contains_backend_section():
    analysis = analyze_results(_sample_results())
    report = generate_summary_report(analysis)
    assert "DISTRIBUICAO DE BACKEND" in report
    assert "torch -> torch:1" in report
    assert "jax -> torch:1" in report
