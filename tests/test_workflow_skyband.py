"""Testes do Skyband aplicado a resumos de workflows e tarefas."""

from experiment.workflow import ExperimentRun, TaskExecutionAttempt, TaskRun, TaskStatus
from gridsearch.workflow_skyband import task_skyband_query, workflow_skyband_query


def _workflow(run_id: str, prepare_time: float, train_time: float) -> ExperimentRun:
    def task(task_id: str, duration: float, cost: float, f1_score: float) -> TaskRun:
        return TaskRun(
            task_id, task_id, "train", TaskStatus.SUCCEEDED,
            [TaskExecutionAttempt(
                f"{run_id}-{task_id}", 1, TaskStatus.SUCCEEDED,
                metrics={
                    "resources": {"task_time_sec": duration, "cost_usd": cost},
                    "evaluation": {"f1_score": f1_score},
                },
            )],
        )

    return ExperimentRun(
        run_id, "benchmark", "success",
        [
            task("prepare", prepare_time, 1.0, 0.0),
            task("train", train_time, 2.0, 0.9 if run_id == "run-a" else 0.8),
        ],
    )


def test_workflow_skyband_projects_aggregated_resource_and_quality_metrics():
    workflows = [_workflow("run-a", 3.0, 7.0), _workflow("run-b", 4.0, 9.0)]

    frontier = workflow_skyband_query(
        workflows,
        k=1,
        metrics=["task_time_sec", "cost_usd", "f1_score"],
        minimize=[True, True, False],
    )

    assert [point["experiment_run_id"] for point in frontier] == ["run-a"]
    assert frontier[0]["resources"]["task_time_sec"] == 10.0
    assert frontier[0]["resources"]["cost_usd"] == 3.0
    assert frontier[0]["evaluation"]["f1_score"] == 0.9


def test_task_skyband_compares_only_the_requested_task_across_workflows():
    workflows = [_workflow("run-a", 10.0, 8.0), _workflow("run-b", 2.0, 4.0)]
    workflows[1].tasks[1].status = TaskStatus.CACHED
    workflows[1].tasks[1].attempts[0].status = TaskStatus.CACHED

    frontier = task_skyband_query(
        workflows,
        "train",
        k=1,
        metrics=["task_time_sec", "cost_usd"],
        minimize=[True, True],
    )

    assert [point["experiment_run_id"] for point in frontier] == ["run-b"]
    assert frontier[0]["task_id"] == "train"
    assert frontier[0]["resources"]["task_time_sec"] == 4.0