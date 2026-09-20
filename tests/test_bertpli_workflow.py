"""Testes do workflow BERT-PLI de referência."""

from experiment import bertpli_workflow
from experiment.bertpli_workflow import (
    BertPliWorkflowConfig,
    build_bertpli_task_functions,
    build_bertpli_workflow,
)
from experiment.task_executor import SequentialWorkflowExecutor
from experiment.workflow import ExecutionRegime, TaskActivity


def test_bertpli_workflow_defines_expected_dag():
    workflow = build_bertpli_workflow(BertPliWorkflowConfig())
    tasks = {task.task_id: task for task in workflow.tasks}

    assert workflow.experiment_type == "nlp"
    assert len(tasks) == 7
    assert tasks["poolout"].depends_on == ("fine_tune_bert",)
    assert tasks["train_attention_rnn"].depends_on == (
        "convert_poolout_train", "convert_poolout_valid"
    )
    assert tasks["evaluate_retrieval"].depends_on == ("test_attention_rnn",)


def test_bertpli_workflow_tasks_classified_by_activity_and_regime():
    """BL-W1: cada tarefa deve refletir a taxonomia T0-T5 do template de workflows."""
    tasks = {task.task_id: task for task in build_bertpli_workflow(BertPliWorkflowConfig()).tasks}

    expected_activity = {
        "fine_tune_bert": TaskActivity.ADAPTATION,
        "poolout": TaskActivity.INGESTION,
        "convert_poolout_train": TaskActivity.INGESTION,
        "convert_poolout_valid": TaskActivity.INGESTION,
        "train_attention_rnn": TaskActivity.ADAPTATION,
        "test_attention_rnn": TaskActivity.EVALUATION_MONITORING,
        "evaluate_retrieval": TaskActivity.EVALUATION_MONITORING,
    }
    for task_id, activity in expected_activity.items():
        assert tasks[task_id].activity == activity, task_id
        assert tasks[task_id].regime == ExecutionRegime.BUILD, task_id

    # Tarefas de treino (atualizam pesos) sao fortemente acopladas e usam GPU.
    for task_id in ("fine_tune_bert", "train_attention_rnn"):
        assert tasks[task_id].resources.gpu_count >= 1
        assert tasks[task_id].resources.coupling_degree == 0.9

    # Tarefas puramente de conversao/metricas nao usam GPU nem tem acoplamento.
    for task_id in ("convert_poolout_train", "convert_poolout_valid", "evaluate_retrieval"):
        assert tasks[task_id].resources.gpu_count == 0
        assert tasks[task_id].resources.coupling_degree == 0.0


def test_bertpli_workflow_gpu_count_reflects_config_gpu_string():
    """BL-W1: gpu_count deriva de BertPliWorkflowConfig.gpu (ex.: '0,1' -> 2)."""
    tasks_single = {t.task_id: t for t in build_bertpli_workflow(BertPliWorkflowConfig(gpu="0")).tasks}
    tasks_multi = {t.task_id: t for t in build_bertpli_workflow(BertPliWorkflowConfig(gpu="0,1")).tasks}
    tasks_auto = {t.task_id: t for t in build_bertpli_workflow(BertPliWorkflowConfig(gpu=None)).tasks}

    assert tasks_single["fine_tune_bert"].resources.gpu_count == 1
    assert tasks_multi["fine_tune_bert"].resources.gpu_count == 2
    assert tasks_auto["fine_tune_bert"].resources.gpu_count == 1


def test_bertpli_task_adapters_execute_existing_clis_in_workflow_order(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    metrics_result = tmp_path / "metrics.json"
    monkeypatch.setattr(bertpli_workflow, "parse_gru_results", lambda *_args: None)
    monkeypatch.setattr(bertpli_workflow, "compute_metrics", lambda *_args: {"f1_score": 0.9})
    config = BertPliWorkflowConfig(gpu="0", metrics_result=str(metrics_result))
    functions = build_bertpli_task_functions(config, command_runner=commands.append)
    workflow = build_bertpli_workflow(config)

    result = SequentialWorkflowExecutor(functions).execute(workflow)

    assert result.status == "success"
    assert [task.task_id for task in result.tasks] == [
        "fine_tune_bert", "poolout", "convert_poolout_train", "convert_poolout_valid",
        "train_attention_rnn", "test_attention_rnn", "evaluate_retrieval",
    ]
    assert len(commands) == 6
    assert commands[0][-2:] == ["--gpu", "0"]
    assert "scripts.poolout" in commands[1]
    assert "scripts.poolout_to_train" in commands[2]
    assert metrics_result.exists()


def test_bertpli_training_adapters_attach_existing_profiling_metrics(tmp_path):
    config_path = tmp_path / "model.config"
    config_path.write_text("[output]\nmodel_path = " + str(tmp_path) + "\nmodel_name = profile\n")
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "profiling_metrics.json").write_text(
        '{"total_gflops": 42.0, "avg_gflops_per_batch": 7.0}', encoding="utf-8"
    )
    config = BertPliWorkflowConfig(bert_config=str(config_path), rnn_config=str(config_path))
    functions = build_bertpli_task_functions(config, command_runner=lambda _command: None)

    fine_tune = functions["fine_tune_bert"]()
    train_rnn = functions["train_attention_rnn"]()

    assert fine_tune["metrics"]["resources"]["total_gflops"] == 42.0
    assert train_rnn["metrics"]["resources"]["avg_gflops_per_batch"] == 7.0