"""Integração do workflow BERT-PLI com o CLI."""

from cli.commands import BertPliWorkflowCommand, _resolve_command
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