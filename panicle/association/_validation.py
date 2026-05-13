"""Validation helpers shared by association methods."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _format_labels(labels: np.ndarray, limit: int = 10) -> str:
    shown = [str(label) for label in labels[:limit]]
    suffix = ""
    if labels.size > limit:
        suffix = f", ... and {labels.size - limit} more"
    return ", ".join(shown) + suffix


def describe_invalid_samples(
    invalid_mask: np.ndarray,
    *,
    sample_ids: Optional[Sequence[object]] = None,
    label: str = "sample",
    limit: int = 10,
) -> str:
    """Return a compact description of invalid rows for error messages."""
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    invalid_rows = np.flatnonzero(invalid_mask)
    if invalid_rows.size == 0:
        return "0 rows"

    if sample_ids is None:
        labels = invalid_rows.astype(object)
        label_text = "row indices"
    else:
        ids = np.asarray(sample_ids, dtype=object)
        if ids.shape[0] != invalid_mask.shape[0]:
            labels = invalid_rows.astype(object)
            label_text = "row indices"
        else:
            labels = ids[invalid_rows]
            label_text = f"{label} IDs"

    return f"{invalid_rows.size} rows; affected {label_text}: {_format_labels(labels, limit=limit)}"


def missing_values_error(
    subject: str,
    invalid_mask: np.ndarray,
    *,
    sample_ids: Optional[Sequence[object]] = None,
    action: str,
    label: str = "sample",
) -> ValueError:
    """Build a ValueError that identifies rows with missing/non-finite values."""
    details = describe_invalid_samples(invalid_mask, sample_ids=sample_ids, label=label)
    return ValueError(f"{subject} contains missing/non-finite values ({details}); {action}")
