"""SYNTHETIC aggregate benchmark provenance and validator tests."""

from cardiosentinel.data.models import (
    AnnotationSample,
    DatasetRecord,
    ParsedAnnotations,
    STEvent,
)
from cardiosentinel.evaluation.benchmark import summarize_benchmark
from cardiosentinel.evaluation.splits import split_sha256


def synthetic_record(index: int) -> DatasetRecord:
    return DatasetRecord(
        "ltstdb",
        "1.0.0",
        f"s900{index}1",
        f"ltstdb:s900{index}",
        10.0,
        ("I",),
        ("I",),
        1,
        300,
        30.0,
        "SYNTHETIC",
    )


def synthetic_annotations(record: DatasetRecord) -> ParsedAnnotations:
    raw = AnnotationSample(0, "s", aux_note="SYNTHETIC")
    event = STEvent(
        "ltstdb",
        record.record_id,
        record.subject_id,
        0,
        "st_episode",
        "ischemic",
        0,
        150,
        300,
        0.0,
        15.0,
        30.0,
        100.0,
        "elevation",
        "stb",
        "ltstdb.stb",
        True,
        (raw,),
    )
    return ParsedAnnotations((event,), (), ())


def synthetic_split(records: tuple[DatasetRecord, ...]) -> dict[str, object]:
    manifest: dict[str, object] = {
        "protocol_version": "1",
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "annotation_definition": "ltstdb.stb",
        "sealed_test_partition": True,
        "partitions": {
            "train": {"subjects": [records[0].subject_id]},
            "validation": {"subjects": [records[1].subject_id]},
            "test": {"subjects": [records[2].subject_id]},
        },
        "records_by_subject": {
            record.subject_id: [record.record_id] for record in records
        },
    }
    manifest["split_sha256"] = split_sha256(manifest)
    return manifest


def test_summary_has_protocol_provenance_and_all_partitions() -> None:
    records = tuple(synthetic_record(index) for index in range(3))
    parsed = tuple(synthetic_annotations(record) for record in records)
    split = synthetic_split(records)
    summary = summarize_benchmark(
        records,
        parsed,
        dataset_id="ltstdb",
        annotation_definition="ltstdb.stb",
        command="SYNTHETIC benchmark command",
        split=split,
        expected_split_hash=split["split_sha256"],
        expected_subject_count=3,
        expected_record_count=3,
    )
    assert set(summary["partitions"]) == {"train", "validation", "test"}
    assert all(
        bucket["window_counts"]["ischemic_positive"] > 0
        for bucket in summary["partitions"].values()
    )
    provenance = summary["provenance"]
    assert provenance["benchmark_schema_version"] == "3.0"
    assert provenance["benchmark_protocol_version"] == "1"
    assert provenance["label_policy_version"] == "1"
    assert provenance["generation_command"] == "SYNTHETIC benchmark command"
