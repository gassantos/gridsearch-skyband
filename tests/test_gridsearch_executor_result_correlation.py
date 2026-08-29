"""Testa a correlação direta entre worker de grid e seu resultado."""

from gridsearch.executor import run_single_experiment


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
    assert result["status"] == "success"