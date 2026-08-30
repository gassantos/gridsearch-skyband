"""Integracao dos workflows declarativos com o CLI."""

import json

import pytest

from cli.commands import (
    BertPliWorkflowCommand,
    GenericWorkflowCommand,
    _resolve_command,
)
from cli.parser import build_argument_parser


def test_parser_resolves_bertpli_workflow_command():
    args = build_argument_parser().parse_args(["--workflow", "bertpli", "--workflow-dry-run"])

    assert args.workflow_dry_run is True
    assert isinstance(_resolve_command(args), BertPliWorkflowCommand)


def test_bertpli_dry_run_persists_workflow(monkeypatch, tmp_path, capsys):
    workflows = []
    monkeypatch.setattr(
        "cli.commands.write_workflow_run",
        lambda workflow: workflows.append(workflow) or tmp_path / workflow.experiment_run_id,
    )
    args = build_argument_parser().parse_args(["--workflow", "bertpli", "--workflow-dry-run"])

    BertPliWorkflowCommand().execute(args, {})

    assert workflows[0].status == "success"
    assert "Workflow BERT-PLI validado sem treinamento" in capsys.readouterr().out


def test_generic_workflow_dry_run_persists_multidomain_spec(monkeypatch, tmp_path, capsys):
    spec_path = tmp_path / "workflow.json"
    spec_path.write_text(json.dumps({
        "name": "classic", "experiment_type": "ml_classic",
        "tasks": [{"task_id": "train", "name": "Treinar", "command": ["python", "train.py"]}],
    }), encoding="utf-8")
    workflows = []
    monkeypatch.setattr(
        "cli.commands.write_workflow_run",
        lambda workflow: workflows.append(workflow) or tmp_path / workflow.experiment_run_id,
    )
    args = build_argument_parser().parse_args(
        ["--workflow", "generic", "--workflow-spec", str(spec_path), "--workflow-dry-run"]
    )

    assert isinstance(_resolve_command(args), GenericWorkflowCommand)
    GenericWorkflowCommand().execute(args, {})

    assert workflows[0].status == "success"
    assert "Workflow generico validado" in capsys.readouterr().out


def test_generic_workflow_requires_specification_file():
    args = build_argument_parser().parse_args(["--workflow", "generic"])

    with pytest.raises(ValueError, match="workflow-spec"):
        GenericWorkflowCommand().execute(args, {})