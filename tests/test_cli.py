from typer.testing import CliRunner

from drybean.cli import app

runner = CliRunner()


def test_cli_exposes_required_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "analyze",
        "train",
        "ablation",
        "robustness",
        "plot",
        "all",
    ):
        assert command in result.stdout

