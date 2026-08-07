"""Pre-specified constants for benchmark protocol V1."""

from __future__ import annotations

from typing import Final

from cardiosentinel.evaluation.models import ChallengeEvidencePolicy, ChallengeName

BENCHMARK_SCHEMA_VERSION: Final = "3.0"
PROTOCOL_VERSION: Final = "1"
LABEL_POLICY_VERSION: Final = "1"
PRIMARY_DATASET: Final = "ltstdb"
PRIMARY_DATASET_VERSION: Final = "1.0.0"
PRIMARY_ANNOTATION_DEFINITION: Final = "ltstdb.stb"
PRIMARY_WINDOW_SECONDS: Final = 10.0
PRIMARY_STRIDE_SECONDS: Final = 5.0
MARKER_VICINITY_SECONDS: Final = 30.0
DEFAULT_SEED: Final = 2026
TRAIN_NEGATIVE_RATIO: Final = 3
NO_POSITIVE_SUBJECT_NEGATIVE_CAP: Final = 30
BOOTSTRAP_REPLICATES: Final = 1000
BOOTSTRAP_SEED: Final = 2026

CHALLENGE_EVIDENCE_POLICIES: Final = {
    "rate_related": ChallengeEvidencePolicy(
        challenge="rate_related",
        target_family="rate_related_confounder",
        evidence_level="quantitative_secondary",
        is_headline_metric=False,
        supports_inferential_bootstrap=True,
    ),
    "axis_shift": ChallengeEvidencePolicy(
        challenge="axis_shift",
        target_family="axis_shift_confounder",
        evidence_level="quantitative_secondary",
        is_headline_metric=False,
        supports_inferential_bootstrap=True,
    ),
    "conduction_change": ChallengeEvidencePolicy(
        challenge="conduction_change",
        target_family="conduction_change_confounder",
        evidence_level="exploratory_descriptive",
        is_headline_metric=False,
        supports_inferential_bootstrap=False,
    ),
}
POSITIVE_CONTEXT_EVIDENCE_LEVEL: Final = "descriptive_error_analysis"
POSITIVE_CONTEXT_FLAGS: Final = (
    "axis_shift_context",
    "conduction_change_context",
    "point_noise_context",
)


def challenge_evidence_policy(challenge: ChallengeName) -> ChallengeEvidencePolicy:
    """Return the V1 interpretation policy for one challenge subset."""
    return CHALLENGE_EVIDENCE_POLICIES[challenge]

LTSTDB_V1_SPLIT_SHA256: Final = (
    "66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7"
)
LTSTDB_V1_GENERATOR_CODE_SHA256: Final = (
    "0211c84004ff5dd97d8e6e99418d5c779d65d18fe4f05bc807415211bb2db52d"
)
