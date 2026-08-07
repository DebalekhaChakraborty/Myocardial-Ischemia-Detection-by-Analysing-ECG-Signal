import hashlib
import json
from pathlib import Path

import pytest

from cardiosentinel.cli import main
from cardiosentinel.neural.artifacts import ProspectiveB4RunRecord
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_PATH,
    B4_PROTOCOL_SHA256,
    validate_frozen_protocol,
)
from cardiosentinel.neural.provenance import validate_source_verification_receipt


def test_frozen_protocol_digest_is_captured() -> None:
    digest = hashlib.sha256(B4_PROTOCOL_PATH.read_bytes()).hexdigest()
    assert digest == B4_PROTOCOL_SHA256
    assert validate_frozen_protocol() == B4_PROTOCOL_SHA256


def test_source_receipt_validation_without_waveform_access(tmp_path: Path) -> None:
    source = tmp_path.resolve()
    receipt = {
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "expected_record_count": 86,
        "verified_required_file_count": 259,
        "verification_result": "passed",
        "local_source_root": str(source),
        "official_manifest": {
            "identifier": "https://physionet.org/files/ltstdb/1.0.0/SHA256SUMS.txt",
            "sha256": (
                "88b8c6d17451d758defb3355526430d607f0eeccf7fc9bb7dc992cde212c220e"
            ),
        },
    }
    (source / "source_verification.json").write_text(json.dumps(receipt))

    validated = validate_source_verification_receipt(source)
    assert validated["official_manifest"] == receipt["official_manifest"]
    assert validated["resolved_local_source_root"] == str(source)


def test_b4_cli_has_no_test_evaluation_command(capsys) -> None:
    with pytest.raises(SystemExit) as result:
        main(["b4", "--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "preflight" in output
    assert "smoke" in output
    assert "benchmark-io" in output
    assert "evaluate-test" not in output


def test_prospective_artifact_contract_keeps_test_null() -> None:
    record = ProspectiveB4RunRecord(
        protocol_sha256="a" * 64,
        source_provenance={},
        split_sha256="b" * 64,
        training_selection_sha256="c" * 64,
        git_sha="d" * 40,
        git_dirty=False,
        environment={},
        device="cpu",
        model_config={},
        trainable_parameter_count=87_089,
        seed=2026,
        epoch_history=(),
        selected_checkpoint_epoch=None,
        validation_auprc=None,
        threshold=None,
        checkpoint_sha256=None,
        checkpoint_serialized_bytes=None,
        timings={},
    )
    assert record.test is None
