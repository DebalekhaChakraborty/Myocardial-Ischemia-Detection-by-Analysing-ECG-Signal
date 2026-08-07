"""Deterministic subject-aware training sampling without model invocation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

from cardiosentinel.evaluation.models import WindowTarget
from cardiosentinel.evaluation.protocol import (
    DEFAULT_SEED,
    NO_POSITIVE_SUBJECT_NEGATIVE_CAP,
    TRAIN_NEGATIVE_RATIO,
)


def _seeded_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def sample_training_targets(
    targets: Iterable[WindowTarget],
    *,
    partition: str,
    seed: int = DEFAULT_SEED,
    negative_ratio: int = TRAIN_NEGATIVE_RATIO,
    no_positive_subject_cap: int = NO_POSITIVE_SUBJECT_NEGATIVE_CAP,
) -> tuple[WindowTarget, ...]:
    """Keep all positives and draw at most 3x distributed background negatives."""
    if partition != "train":
        raise ValueError("Negative sampling is permitted only for training data.")
    if negative_ratio < 0 or no_positive_subject_cap < 0:
        raise ValueError("Sampling limits cannot be negative.")
    targets = tuple(targets)
    positives = tuple(
        target
        for target in targets
        if target.eligible_for_training
        and target.target_family == "ischemic_positive"
    )
    negatives = tuple(
        target
        for target in targets
        if target.eligible_for_training
        and target.target_family == "background_negative"
    )
    budget = min(len(negatives), negative_ratio * len(positives))
    positive_subjects = {target.subject_id for target in positives}
    grouped: dict[tuple[str, str], list[WindowTarget]] = defaultdict(list)
    for target in negatives:
        grouped[(target.subject_id, target.record_id)].append(target)
    for group in grouped.values():
        group.sort(key=lambda item: _seeded_order(item.stable_id, seed))
    group_keys = sorted(
        grouped,
        key=lambda item: _seeded_order(f"{item[0]}:{item[1]}", seed),
    )
    selected: list[WindowTarget] = []
    selected_ids: set[str] = set()
    no_positive_counts: dict[str, int] = defaultdict(int)

    def select_round(keys: Iterable[tuple[str, str]], zero_subjects_only: bool) -> bool:
        progressed = False
        for key in keys:
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
            while grouped[key] and grouped[key][0].stable_id in selected_ids:
                grouped[key].pop(0)
            if not grouped[key]:
                continue
            target = grouped[key].pop(0)
            selected.append(target)
            selected_ids.add(target.stable_id)
            if subject not in positive_subjects:
                no_positive_counts[subject] += 1
            progressed = True
        return progressed

    while len(selected) < budget and select_round(group_keys, True):
        pass
    while len(selected) < budget and select_round(group_keys, False):
        pass
    return tuple(sorted((*positives, *selected), key=lambda item: item.stable_id))


def primary_evaluation_targets(
    targets: Iterable[WindowTarget], *, partition: str
) -> tuple[WindowTarget, ...]:
    """Return the full eligible validation/test composition without downsampling."""
    if partition not in {"validation", "test"}:
        raise ValueError("Full primary evaluation applies to validation or test.")
    return tuple(target for target in targets if target.eligible_for_primary_evaluation)
