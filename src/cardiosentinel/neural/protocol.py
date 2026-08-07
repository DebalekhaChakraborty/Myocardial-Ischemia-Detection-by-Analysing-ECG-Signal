"""Machine-checkable constants from the frozen prospective B4 protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
B4_PROTOCOL_PATH = REPOSITORY_ROOT / "docs" / "B4_PROTOCOL_V1.md"
B4_PROTOCOL_SHA256 = (
    "f6f5e9ed728c86a9b2bd75b2327b9199f0e097b91387525a192c212e6771b28b"
)
B4_SPLIT_SHA256 = (
    "66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7"
)
FEATURE_CORPUS_SHA256 = (
    "f18785d520828cb171482926922346dda824c8868ed4b7f9be45897cd71d6eb5"
)
DATASET = "ltstdb"
DATASET_VERSION = "1.0.0"
SAMPLING_FREQUENCY_HZ = 250.0
WINDOW_SAMPLES = 2_500
WINDOW_SECONDS = 10.0
STRIDE_SECONDS = 5.0
SEED = 2026
TRAINABLE_PARAMETER_COUNT = 87_089
FP32_PARAMETER_BYTES = 348_356
LOCAL_RECEPTIVE_FIELD_SAMPLES = 1_943
TEMPORAL_LENGTHS = (2_500, 1_250, 625, 313, 157, 79)
PRIMARY_FAMILIES = frozenset(("ischemic_positive", "background_negative"))
EXPECTED_COUNTS = {
    "train": {
        "positive": 93_613,
        "negative": 280_839,
        "total": 374_452,
        "subjects": 56,
    },
    "validation": {
        "positive": 21_628,
        "negative": 452_269,
        "total": 473_897,
    },
}

DevelopmentPartition = Literal["train", "validation"]


def require_development_partition(partition: str) -> DevelopmentPartition:
    """Reject sealed test access before callers resolve any artifact path."""
    if partition not in {"train", "validation"}:
        raise ValueError("B4 development permits train and validation only.")
    return cast(DevelopmentPartition, partition)


def protocol_sha256(path: Path = B4_PROTOCOL_PATH) -> str:
    """Hash the exact protocol bytes without normalizing the document."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_protocol(path: Path = B4_PROTOCOL_PATH) -> str:
    """Fail if the prospective protocol bytes differ from the frozen digest."""
    digest = protocol_sha256(path)
    if digest != B4_PROTOCOL_SHA256:
        raise ValueError("B4_PROTOCOL_V1.md differs from its frozen SHA-256.")
    return digest
