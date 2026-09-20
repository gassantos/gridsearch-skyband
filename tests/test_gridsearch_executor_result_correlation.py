"""Testa a correlação direta entre worker de grid e seu resultado."""

import json

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


def test_run_single_experiment_attaches_huggingface_workflow_metadata(monkeypatch):
    def fake_launch_experiment(**_kwargs):
        return {"experiment": {"id": "worker-result"}, "resources": {}}

    monkeypatch.setattr("experiment.xla_launcher.launch_experiment", fake_launch_experiment)

    result = run_single_experiment(
        experiment_idx=7,
        config_path="ignored.config",
        params={"learning_rate": 2e-5, "environment": "local"},
        dataset_overrides={"hf_dataset_source": "hub", "hf_dataset_id": "nyu-mll/glue"},
        environment_details={"gpu": "NVIDIA RTX 3090", "vram_gb": 24},
        train_dataset="train_task2_v3",
    )
    tasks = {task["task_id"]: task for task in result["workflow"]["tasks"]}

    assert result["workflow"]["experiment_type"] == "llm"
    assert tasks["ingest_dataset"]["activity"] == "ingestion"
    assert tasks["adapt_model"]["activity"] == "adaptation"
    assert tasks["evaluate_model"]["activity"] == "evaluation_monitoring"
    assert tasks["ingest_dataset"]["outputs"][0]["uri"] == "hf://datasets/nyu-mll/glue"
    assert tasks["ingest_dataset"]["config"]["train_dataset"] == "train_task2_v3"
    assert tasks["adapt_model"]["resources"]["gpu_count"] == 1
    assert json.dumps(result["workflow"])


def test_run_single_experiment_keeps_workflow_metadata_when_execution_fails(monkeypatch):
    def fake_launch_experiment(**_kwargs):
        raise RuntimeError("launcher failure")

    monkeypatch.setattr("experiment.xla_launcher.launch_experiment", fake_launch_experiment)

    result = run_single_experiment(
        experiment_idx=3,
        config_path="ignored.config",
        params={"environment": "local"},
        dataset_overrides={"hf_dataset_source": "hub", "hf_dataset_id": "nyu-mll/glue"},
    )

    assert result["status"] == "failed"
    assert result["workflow"]["name"] == "grid-experiment-3"
    assert result["workflow"]["tasks"][0]["outputs"][0]["uri"] == "hf://datasets/nyu-mll/glue"


def test_grid_state_persists_workflow_and_resume_keeps_it(monkeypatch, tmp_path):
    calls = []

    def fake_launch_experiment(**_kwargs):
        calls.append("launch")
        return {"experiment": {"id": "worker-result"}, "resources": {}}

    monkeypatch.setattr("experiment.xla_launcher.launch_experiment", fake_launch_experiment)
    monkeypatch.setattr(
        executor_mod,
        "create_config_for_combination",
        lambda *_args, **_kwargs: "ignored.config",
    )
    grid_config = {
        "hyperparameters": {"learning_rate": [2e-5]},
        "environments": {
            "active": ["local"],
            "details": {"local": {"gpu": "NVIDIA RTX 3090", "vram_gb": 24}},
        },
    }

    run_grid_search(
        base_config_path="ignored.config",
        grid_config=grid_config,
        dataset_overrides={"hf_dataset_source": "hub", "hf_dataset_id": "nyu-mll/glue"},
        output_dir=tmp_path,
    )
    state_path = next(tmp_path.glob("grid_search_state_*.json"))
    persisted = json.loads(state_path.read_text(encoding="utf-8"))

    assert calls == ["launch"]
    assert persisted["results"][0]["workflow"]["tasks"][0]["activity"] == "ingestion"

    run_grid_search(
        base_config_path="ignored.config",
        grid_config=grid_config,
        dataset_overrides={"hf_dataset_source": "hub", "hf_dataset_id": "nyu-mll/glue"},
        resume=True,
        output_dir=tmp_path,
    )
    resumed = json.loads(state_path.read_text(encoding="utf-8"))

    assert calls == ["launch"]
    assert resumed["results"] == persisted["results"]


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


def test_run_grid_search_persists_resource_catalog_when_environments_present(monkeypatch, tmp_path):
    """BL-MILP: o estagio de coleta de recursos roda uma vez, antes dos experimentos."""
    def fake_create_config_for_combination(_base_config_path, _params, idx, **_kwargs):
        return f"fake_config_{idx}.config"

    def fake_run_single_experiment(experiment_idx, _config_path, params, gpu_list=None, **_kwargs):
        return {"status": "success", "grid_experiment_idx": experiment_idx, "grid_params": params}

    monkeypatch.setattr(executor_mod, "create_config_for_combination", fake_create_config_for_combination)
    monkeypatch.setattr(executor_mod, "run_single_experiment", fake_run_single_experiment)

    grid_config = {
        "hyperparameters": {"learning_rate": [1e-5]},
        "environments": {
            "active": ["local"],
            "details": {"local": {"cost_per_hour_usd": 0.04}},
        },
    }

    run_grid_search(
        base_config_path="ignored.config", grid_config=grid_config,
        parallel=1, gpu_ids=[0], output_dir=tmp_path,
    )

    catalog_files = list(tmp_path.glob("resource_catalog_*.json"))
    assert len(catalog_files) == 1
    persisted = json.loads(catalog_files[0].read_text(encoding="utf-8"))
    assert "local" in persisted["resources"]
    assert "communication_costs" in persisted


def test_run_grid_search_skips_resource_catalog_without_environments(tmp_path, monkeypatch):
    """BL-MILP: sem dimensao environments, nenhum catalogo e persistido."""
    def fake_create_config_for_combination(_base_config_path, _params, idx, **_kwargs):
        return f"fake_config_{idx}.config"

    def fake_run_single_experiment(experiment_idx, _config_path, params, gpu_list=None, **_kwargs):
        return {"status": "success", "grid_experiment_idx": experiment_idx, "grid_params": params}

    monkeypatch.setattr(executor_mod, "create_config_for_combination", fake_create_config_for_combination)
    monkeypatch.setattr(executor_mod, "run_single_experiment", fake_run_single_experiment)

    grid_config = {"hyperparameters": {"learning_rate": [1e-5]}}

    run_grid_search(
        base_config_path="ignored.config", grid_config=grid_config,
        parallel=1, gpu_ids=[0], output_dir=tmp_path,
    )

    assert list(tmp_path.glob("resource_catalog_*.json")) == []