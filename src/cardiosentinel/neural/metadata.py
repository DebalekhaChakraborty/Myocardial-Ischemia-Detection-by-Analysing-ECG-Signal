"""Metadata-only B4 window indexing with a hard sealed-test firewall."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from cardiosentinel.baseline.cache import (
    FEATURE_MANIFEST_NAME,
    read_json,
    require_nonversioned_path,
)
from cardiosentinel.baseline.sampling import build_training_selection_plan
from cardiosentinel.neural.protocol import (
    B4_SPLIT_SHA256,
    DATASET,
    DATASET_VERSION,
    EXPECTED_COUNTS,
    FEATURE_CORPUS_SHA256,
    PRIMARY_FAMILIES,
    DevelopmentPartition,
    require_development_partition,
)


@dataclass(frozen=True, slots=True)
class B4WindowReference:
    """Non-predictive identity and target metadata for one raw B4 window."""

    stable_id: str
    record_id: str
    subject_id: str
    channel_index: int
    start_sample: int
    end_sample: int
    partition: DevelopmentPartition
    target_family: str
    context_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        require_development_partition(self.partition)
        if self.channel_index < 0 or self.start_sample < 0:
            raise ValueError("B4 window identity contains a negative index.")
        if self.end_sample <= self.start_sample:
            raise ValueError("B4 window interval must be non-empty.")
        expected = (
            f"{DATASET}:{self.record_id}:{self.channel_index}:"
            f"{self.start_sample}:{self.end_sample}"
        )
        if self.stable_id != expected:
            raise ValueError(
                "B4 stable ID does not match record/channel/window identity."
            )

    @property
    def binary_label(self) -> float:
        if self.target_family not in PRIMARY_FAMILIES:
            raise ValueError("Only primary B4 rows have a binary training label.")
        return float(self.target_family == "ischemic_positive")


@dataclass(frozen=True, slots=True)
class B4MetadataIndex:
    partition: DevelopmentPartition
    references: tuple[B4WindowReference, ...]
    positive_count: int
    negative_count: int
    subject_count: int
    selection_sha256: str | None = None

    @property
    def total_count(self) -> int:
        return len(self.references)


def canonical_selection_sha256(stable_ids: Iterable[str]) -> str:
    """Hash a canonical JSON array of unique, sorted selected stable IDs."""
    identifiers = tuple(stable_ids)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Training selection contains duplicate stable IDs.")
    payload = json.dumps(sorted(identifiers), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metadata_arrays(path: Path) -> tuple[NDArray, ...]:
    """Read only B4 metadata members; never request the `features` NPZ member."""
    with np.load(path, allow_pickle=False) as cached:
        return (
            np.asarray(cached["stable_ids"], dtype=np.str_),
            np.asarray(cached["record_ids"], dtype=np.str_),
            np.asarray(cached["subject_ids"], dtype=np.str_),
            np.asarray(cached["channel_indices"], dtype=np.int64),
            np.asarray(cached["window_start_samples"], dtype=np.int64),
            np.asarray(cached["window_end_samples"], dtype=np.int64),
            np.asarray(cached["partitions"], dtype=np.str_),
            np.asarray(cached["target_families"], dtype=np.str_),
            np.asarray(cached["context_flags"], dtype=np.str_),
        )


def _manifest_identity(manifest: dict) -> None:
    expected = {
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "processing_profile": "raw",
        "window_seconds": 10.0,
        "stride_seconds": 5.0,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("B4 metadata corpus differs from the frozen identity.")


def _partition_entries(
    feature_root: Path, partition: str
) -> tuple[Path, DevelopmentPartition, tuple[dict, ...]]:
    # The firewall deliberately runs before root resolution or manifest access.
    permitted = require_development_partition(partition)
    root = require_nonversioned_path(feature_root, "B4 feature metadata root")
    manifest = read_json(root / FEATURE_MANIFEST_NAME)
    _manifest_identity(manifest)
    entries = tuple(
        sorted(
            (
                entry
                for entry in manifest.get("records", ())
                if entry.get("partition") == permitted
                and entry.get("status") == "complete"
            ),
            key=lambda entry: str(entry["record_id"]),
        )
    )
    if not entries:
        raise ValueError(f"B4 metadata has no complete {permitted} records.")
    return root, permitted, entries


def load_b4_references(
    feature_root: Path,
    partition: str,
    *,
    primary_only: bool = True,
) -> tuple[B4WindowReference, ...]:
    """Load train/validation identity arrays without numeric feature vectors."""
    root, permitted, entries = _partition_entries(feature_root, partition)
    references: list[B4WindowReference] = []
    for entry in entries:
        cache_path = (root / str(entry["cache_path"])).resolve()
        try:
            cache_path.relative_to(root)
        except ValueError as error:
            raise ValueError("B4 metadata cache path escapes its root.") from error
        arrays = _metadata_arrays(cache_path)
        sizes = {array.size for array in arrays}
        if len(sizes) != 1:
            raise ValueError("B4 metadata arrays are not row-aligned.")
        stable, records, subjects, channels, starts, ends, parts, families, contexts = (
            arrays
        )
        for values in zip(
            stable,
            records,
            subjects,
            channels,
            starts,
            ends,
            parts,
            families,
            contexts,
            strict=True,
        ):
            (
                stable_id,
                record_id,
                subject_id,
                channel_index,
                start,
                end,
                row_partition,
                family,
                context,
            ) = values
            if str(row_partition) != permitted:
                raise ValueError("B4 metadata row has the wrong partition.")
            if primary_only and str(family) not in PRIMARY_FAMILIES:
                continue
            references.append(
                B4WindowReference(
                    stable_id=str(stable_id),
                    record_id=str(record_id),
                    subject_id=str(subject_id),
                    channel_index=int(channel_index),
                    start_sample=int(start),
                    end_sample=int(end),
                    partition=permitted,
                    target_family=str(family),
                    context_flags=tuple(
                        item for item in str(context).split("|") if item
                    ),
                )
            )
    identifiers = [reference.stable_id for reference in references]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("B4 metadata contains duplicate stable IDs.")
    return tuple(sorted(references, key=lambda reference: reference.stable_id))


def build_training_index(feature_root: Path) -> B4MetadataIndex:
    """Reproduce the exact frozen B1-B3 train selection using train metadata only."""
    root = require_nonversioned_path(feature_root, "B4 feature metadata root")
    plan = build_training_selection_plan(root, sampled=True, seed=2026)
    expected = EXPECTED_COUNTS["train"]
    observed_plan = {
        "positive": plan.selected_positive_count,
        "negative": plan.selected_negative_count,
        "total": plan.selected_count,
        "subjects": plan.selected_subject_count,
    }
    if observed_plan != expected:
        raise ValueError(
            f"B4 training selection differs from frozen counts: {observed_plan}"
        )
    references = load_b4_references(root, "train")
    selected = tuple(
        reference
        for reference in references
        if reference.stable_id in plan.selected_stable_ids
    )
    if {reference.stable_id for reference in selected} != set(
        plan.selected_stable_ids
    ):
        raise ValueError(
            "B4 training metadata does not cover every selected stable ID."
        )
    positive = sum(item.target_family == "ischemic_positive" for item in selected)
    negative = sum(item.target_family == "background_negative" for item in selected)
    subjects = len({item.subject_id for item in selected})
    if (positive, negative, len(selected), subjects) != (
        expected["positive"],
        expected["negative"],
        expected["total"],
        expected["subjects"],
    ):
        raise ValueError("B4 selected references differ from frozen train metadata.")
    return B4MetadataIndex(
        partition="train",
        references=selected,
        positive_count=positive,
        negative_count=negative,
        subject_count=subjects,
        selection_sha256=canonical_selection_sha256(plan.selected_stable_ids),
    )


def build_validation_index(feature_root: Path) -> B4MetadataIndex:
    """Expose every primary validation row without sampling."""
    references = load_b4_references(feature_root, "validation")
    positive = sum(item.target_family == "ischemic_positive" for item in references)
    negative = sum(item.target_family == "background_negative" for item in references)
    expected = EXPECTED_COUNTS["validation"]
    if (positive, negative, len(references)) != (
        expected["positive"],
        expected["negative"],
        expected["total"],
    ):
        raise ValueError("B4 validation references differ from frozen metadata.")
    return B4MetadataIndex(
        partition="validation",
        references=references,
        positive_count=positive,
        negative_count=negative,
        subject_count=len({item.subject_id for item in references}),
    )
