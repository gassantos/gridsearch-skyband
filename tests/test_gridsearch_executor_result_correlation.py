"""Testa a correlação direta entre worker de grid e seu resultado."""

import gridsearch.executor as executor_mod
from gridsearch.executor import (
    _environment_capacity_registry,
    run_grid_search,
    run_single_experiment,
)


def test_run_single_experiment_uses_returned_result(monkeypatch):
    expected = {
        "experiment": {"id": "worker-result"},
        "resources": {"train_time_sec": "1.00"},
    }

    def fake_launch_experiment(**_kwargs):
        return expected.copy()

    monkeypatch.setattr(
        "experiment.xla_launcher.launch_experiment",
        fake_launch_experiment,
    )

    result = run_single_experiment(
        experiment_idx=7,
        config_path="ignored.config",
        params={"batch_size": 16},
    )

    assert result["experiment"]["id"] == "worker-result"
    assert result["grid_experiment_idx"] == 7


def test_environment_capacity_registry_extracts_parallel_workers():
    """BL-W2: extrai parallel_workers por ambiente, ignorando entradas invalidas."""
    grid_config = {
        "environments": {
            "details": {
                "colab": {"parallel_workers": 1},
                "local": {"parallel_workers": 2},
                "no_capacity": {"gpu": "some-gpu"},
                "not_a_dict": "oops",
            }
        }
    }

    assert _environment_capacity_registry(grid_config) == {"colab": 1, "local": 2}
    assert _environment_capacity_registry({}) == {}


def test_run_grid_search_respects_environment_capacity_in_gpu_assignment(monkeypatch, tmp_path):
    """BL-W2: round-robin de GPU nao deve exceder a capacidade do ambiente selecionado."""
    calls: list[tuple[int, str, list[int] | None]] = []

    def fake_create_config_for_combination(_base_config_path, _params, idx, **_kwargs):
        return f"fake_config_{idx}.config"

    def fake_run_single_experiment(experiment_idx, _config_path, params, gpu_list=None, **_kwargs):
        calls.append((experiment_idx, params.get("environment"), gpu_list))
        return {"status": "success", "grid_experiment_idx": experiment_idx, "grid_params": params}

    monkeypatch.setattr(executor_mod, "create_config_for_combination", fake_create_config_for_combination)
    monkeypatch.setattr(executor_mod, "run_single_experiment", fake_run_single_experiment)

    grid_config = {
        "hyperparameters": {"learning_rate": [1e-5, 2e-5, 3e-5]},
        "environments": {
            "active": ["colab", "local"],
            "details": {
                "colab": {"parallel_workers": 1},
                "local": {"parallel_workers": 2},
            },
        },
    }

    run_grid_search(
        base_config_path="ignored.config",
        grid_config=grid_config,
        parallel=1,
        gpu_ids=[10, 11, 12, 13],
        output_dir=tmp_path,
    )

    colab_gpus = {tuple(gpu) for _, env, gpu in calls if env == "colab"}
    local_gpus = {tuple(gpu) for _, env, gpu in calls if env == "local"}

    # Colab (capacidade 1) deve sempre usar a mesma GPU do pool, nunca variar.
    assert colab_gpus == {(10,)}
    # Local (capacidade 2) deve ficar restrito as duas primeiras GPUs do pool,
    # nunca usando o restante do pool fisico (12, 13) como o round-robin cego faria.
    assert local_gpus <= {(10,), (11,)}
    assert (12,) not in colab_gpus | local_gpus
    assert (13,) not in colab_gpus | local_gpus