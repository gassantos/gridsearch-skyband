"""Testes do workflow BERT-PLI de referência."""

from experiment import bertpli_workflow
from experiment.bertpli_workflow import (
    BertPliWorkflowConfig,
    build_bertpli_task_functions,
    build_bertpli_workflow,
)
from experiment.task_executor import SequentialWorkflowExecutor


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