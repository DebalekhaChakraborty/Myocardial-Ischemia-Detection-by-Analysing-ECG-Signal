import json

import pytest

from cardiosentinel.cli import main


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as result:
        main(["--help"])

    assert result.value.code == 0
    assert "CardioSentinel research-software utilities." in capsys.readouterr().out


def test_cli_info_is_factual_and_research_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["info"]) == 0

    output = capsys.readouterr().out
    assert "Project: CardioSentinel" in output
    assert "Package version: 0.1.0" in output
    assert "Research software only; not a medical device" in output
    assert "Active configuration profile: base" in output


def test_signal_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as result:
        main(["signal", "--help"])

    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "probe-remote" in output
    assert "filter-audit" in output


def test_benchmark_cli_help_has_no_training_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as result:
        main(["benchmark", "--help"])

    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "summarize" in output
    assert "validate-split" in output
    assert "split-info" in output
    assert "\n    train " not in output


def test_baseline_cli_help_exposes_separate_test_stage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as result:
        main(["baseline", "--help"])

    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "materialize" in output
    assert "preflight" in output
    assert "smoke-remote" in output
    assert "fit" in output
    assert "evaluate-test" in output
    assert "acquire" in output


def test_baseline_acquisition_is_plan_only_without_execute(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "ltstdb" / "1.0.0"
    assert (
        main(
            [
                "baseline",
                "acquire",
                "--destination",
                str(destination),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Long-Term ST Database v1.0.0" in output
    assert "wget" in output
    assert "plan only" in output


def test_raw_filter_audit_is_machine_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["signal", "filter-audit", "--sampling-frequency-hz", "250"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"]["name"] == "raw"
    assert payload["sos_coefficients"] == []
    assert payload["clinical_certification"] is False
