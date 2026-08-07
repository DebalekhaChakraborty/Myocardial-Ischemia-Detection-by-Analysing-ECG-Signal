"""Reference-equivalent index sampling and memory-bounded training loading."""

from __future__ import annotations

import hashlib
import heapq
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cardiosentinel.baseline.cache import (
    FEATURE_MANIFEST_NAME,
    FeatureTable,
    read_feature_table,
    read_json,
)
from cardiosentinel.evaluation.protocol import (
    DEFAULT_SEED,
    NO_POSITIVE_SUBJECT_NEGATIVE_CAP,
    TRAIN_NEGATIVE_RATIO,
)


def _seeded_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def sample_training_indices(
    stable_ids: NDArray[np.str_],
    subject_ids: NDArray[np.str_],
    record_ids: NDArray[np.str_],
    target_families: NDArray[np.str_],
    *,
    seed: int = DEFAULT_SEED,
    negative_ratio: int = TRAIN_NEGATIVE_RATIO,
    no_positive_subject_cap: int = NO_POSITIVE_SUBJECT_NEGATIVE_CAP,
) -> NDArray[np.int64]:
    """Return exactly the canonical sampler's selected rows without target objects."""
    if negative_ratio < 0 or no_positive_subject_cap < 0:
        raise ValueError("Sampling limits cannot be negative.")
    arrays = (stable_ids, subject_ids, record_ids, target_families)
    if len({array.size for array in arrays}) != 1 or any(
        array.ndim != 1 for array in arrays
    ):
        raise ValueError(
            "Training metadata arrays must be one-dimensional and aligned."
        )
    positives = np.flatnonzero(target_families == "ischemic_positive").tolist()
    negatives = np.flatnonzero(target_families == "background_negative").tolist()
    budget = min(len(negatives), negative_ratio * len(positives))
    positive_subjects = {str(subject_ids[index]) for index in positives}
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in negatives:
        grouped[(str(subject_ids[index]), str(record_ids[index]))].append(index)
    for group in grouped.values():
        group.sort(key=lambda index: _seeded_order(str(stable_ids[index]), seed))
    group_keys = sorted(
        grouped,
        key=lambda item: _seeded_order(f"{item[0]}:{item[1]}", seed),
    )
    selected: list[int] = []
    offsets = defaultdict(int)
    no_positive_counts: dict[str, int] = defaultdict(int)

    def select_round(zero_subjects_only: bool) -> bool:
        progressed = False
        for key in group_keys:
            if len(selected) >= budget:
                break
            subject = key[0]
            if zero_subjects_only and subject in positive_subjects:
                continue
            if (
                subject not in positive_subjects
                and no_positive_counts[subject] >= no_positive_subject_cap
            ):
                continue
            offset = offsets[key]
            if offset >= len(grouped[key]):
                continue
            selected.append(grouped[key][offset])
            offsets[key] += 1
            if subject not in positive_subjects:
                no_positive_counts[subject] += 1
            progressed = True
        return progressed

    while len(selected) < budget and select_round(True):
        pass
    while len(selected) < budget and select_round(False):
        pass
    combined = positives + selected
    combined.sort(key=lambda index: str(stable_ids[index]))
    return np.asarray(combined, dtype=np.int64)


@dataclass(frozen=True)
class TrainingSelectionPlan:
    """Selected stable IDs plus provenance counts from metadata-only passes."""

    selected_stable_ids: frozenset[str]
    unsampled_positive_count: int
    unsampled_negative_count: int
    selected_positive_count: int
    selected_negative_count: int
    selected_subject_count: int

    @property
    def unsampled_primary_count(self) -> int:
        return self.unsampled_positive_count + self.unsampled_negative_count

    @property
    def selected_count(self) -> int:
        return self.selected_positive_count + self.selected_negative_count


def _training_entries(feature_root: Path) -> tuple[dict[str, object], ...]:
    manifest = read_json(feature_root / FEATURE_MANIFEST_NAME)
    return tuple(
        sorted(
            (
                entry
                for entry in manifest["records"]
                if entry["partition"] == "train" and entry["status"] == "complete"
            ),
            key=lambda entry: entry["record_id"],
        )
    )


def _metadata_arrays(path: Path) -> tuple[NDArray[np.str_], ...]:
    """Read sampling metadata without decompressing the numeric feature matrix."""
    with np.load(path, allow_pickle=False) as cached:
        return (
            np.asarray(cached["stable_ids"], dtype=np.str_),
            np.asarray(cached["subject_ids"], dtype=np.str_),
            np.asarray(cached["record_ids"], dtype=np.str_),
            np.asarray(cached["target_families"], dtype=np.str_),
        )


def _negative_group_quotas(
    group_counts: dict[tuple[str, str], int],
    positive_subjects: set[str],
    budget: int,
    *,
    seed: int,
    no_positive_subject_cap: int,
) -> dict[tuple[str, str], int]:
    quotas = {key: 0 for key in group_counts}
    group_keys = sorted(
        group_counts,
        key=lambda item: _seeded_order(f"{item[0]}:{item[1]}", seed),
    )
    no_positive_counts: dict[str, int] = defaultdict(int)
    selected = 0

    def select_round(zero_subjects_only: bool) -> bool:
        nonlocal selected
        progressed = False
        for key in group_keys:
            if selected >= budget:
                break
            subject = key[0]
            if zero_subjects_only and subject in positive_subjects:
                continue
            if (
                subject not in positive_subjects
                and no_positive_counts[subject] >= no_positive_subject_cap
            ):
                continue
            if quotas[key] >= group_counts[key]:
                continue
            quotas[key] += 1
            selected += 1
            if subject not in positive_subjects:
                no_positive_counts[subject] += 1
            progressed = True
        return progressed

    while selected < budget and select_round(True):
        pass
    while selected < budget and select_round(False):
        pass
    return quotas


def build_training_selection_plan(
    feature_root: Path,
    *,
    sampled: bool,
    seed: int = DEFAULT_SEED,
    negative_ratio: int = TRAIN_NEGATIVE_RATIO,
    no_positive_subject_cap: int = NO_POSITIVE_SUBJECT_NEGATIVE_CAP,
) -> TrainingSelectionPlan:
    """Scan metadata only; retain bounded selected IDs when sampling is requested."""
    if negative_ratio < 0 or no_positive_subject_cap < 0:
        raise ValueError("Sampling limits cannot be negative.")
    entries = _training_entries(feature_root)
    positive_count = negative_count = 0
    positive_subjects: set[str] = set()
    primary_subjects: set[str] = set()
    positive_ids: set[str] = set()
    group_counts: dict[tuple[str, str], int] = defaultdict(int)
    for entry in entries:
        stable_ids, subject_ids, record_ids, families = _metadata_arrays(
            feature_root / str(entry["cache_path"])
        )
        positive = families == "ischemic_positive"
        negative = families == "background_negative"
        positive_count += int(np.sum(positive))
        negative_count += int(np.sum(negative))
        positive_subjects.update(subject_ids[positive].tolist())
        primary_subjects.update(subject_ids[positive | negative].tolist())
        if sampled:
            positive_ids.update(stable_ids[positive].tolist())
            for subject, record in zip(
                subject_ids[negative], record_ids[negative], strict=True
            ):
                group_counts[(str(subject), str(record))] += 1
    if not sampled:
        return TrainingSelectionPlan(
            selected_stable_ids=frozenset(),
            unsampled_positive_count=positive_count,
            unsampled_negative_count=negative_count,
            selected_positive_count=positive_count,
            selected_negative_count=negative_count,
            selected_subject_count=len(primary_subjects),
        )

    budget = min(negative_count, negative_ratio * positive_count)
    quotas = _negative_group_quotas(
        group_counts,
        positive_subjects,
        budget,
        seed=seed,
        no_positive_subject_cap=no_positive_subject_cap,
    )
    heaps: dict[tuple[str, str], list[tuple[int, str]]] = {
        key: [] for key, quota in quotas.items() if quota
    }
    selected_subjects = set(positive_subjects)
    for entry in entries:
        stable_ids, subject_ids, record_ids, families = _metadata_arrays(
            feature_root / str(entry["cache_path"])
        )
        for stable_id, subject, record in zip(
            stable_ids[families == "background_negative"],
            subject_ids[families == "background_negative"],
            record_ids[families == "background_negative"],
            strict=True,
        ):
            key = (str(subject), str(record))
            quota = quotas.get(key, 0)
            if quota == 0:
                continue
            order = int(_seeded_order(str(stable_id), seed), 16)
            candidate = (-order, str(stable_id))
            heap = heaps[key]
            if len(heap) < quota:
                heapq.heappush(heap, candidate)
            elif order < -heap[0][0]:
                heapq.heapreplace(heap, candidate)
    selected_negative_ids = {
        stable_id for heap in heaps.values() for _, stable_id in heap
    }
    for key, heap in heaps.items():
        if len(heap) != quotas[key]:
            raise ValueError(f"Training sampler could not satisfy group quota {key}.")
        if heap:
            selected_subjects.add(key[0])
    selected_ids = positive_ids | selected_negative_ids
    return TrainingSelectionPlan(
        selected_stable_ids=frozenset(selected_ids),
        unsampled_positive_count=positive_count,
        unsampled_negative_count=negative_count,
        selected_positive_count=positive_count,
        selected_negative_count=len(selected_negative_ids),
        selected_subject_count=len(selected_subjects),
    )


def load_selected_training_table(
    feature_root: Path, selected_stable_ids: frozenset[str]
) -> FeatureTable:
    """Load one record at a time and retain only sampled rows in memory."""
    selected_tables = []
    for entry in _training_entries(feature_root):
        table, _ = read_feature_table(feature_root / str(entry["cache_path"]))
        mask = np.fromiter(
            (stable_id in selected_stable_ids for stable_id in table.stable_ids),
            dtype=np.bool_,
            count=table.row_count,
        )
        if mask.any():
            selected_tables.append(table.select(mask))
    if not selected_tables:
        raise ValueError("Training selection produced no feature rows.")
    table = FeatureTable(
        features=np.vstack([item.features for item in selected_tables]),
        stable_ids=np.concatenate([item.stable_ids for item in selected_tables]),
        record_ids=np.concatenate([item.record_ids for item in selected_tables]),
        subject_ids=np.concatenate([item.subject_ids for item in selected_tables]),
        channel_indices=np.concatenate(
            [item.channel_indices for item in selected_tables]
        ),
        lead_names=np.concatenate([item.lead_names for item in selected_tables]),
        window_start_samples=np.concatenate(
            [item.window_start_samples for item in selected_tables]
        ),
        window_end_samples=np.concatenate(
            [item.window_end_samples for item in selected_tables]
        ),
        partitions=np.concatenate([item.partitions for item in selected_tables]),
        target_families=np.concatenate(
            [item.target_families for item in selected_tables]
        ),
        context_flags=np.concatenate([item.context_flags for item in selected_tables]),
    )
    order = np.argsort(table.stable_ids, kind="stable")
    return FeatureTable(
        features=table.features[order],
        stable_ids=table.stable_ids[order],
        record_ids=table.record_ids[order],
        subject_ids=table.subject_ids[order],
        channel_indices=table.channel_indices[order],
        lead_names=table.lead_names[order],
        window_start_samples=table.window_start_samples[order],
        window_end_samples=table.window_end_samples[order],
        partitions=table.partitions[order],
        target_families=table.target_families[order],
        context_flags=table.context_flags[order],
    )
