"""Genomic inflation rescaled to a common sample size (lambda_1000)."""

from __future__ import annotations

from typing import Optional

import numpy as np


def lambda_1000(
    lambda_gc: float,
    n_samples: Optional[int] = None,
    *,
    n_cases: Optional[int] = None,
    n_controls: Optional[int] = None,
) -> float:
    """Rescale lambda_GC to the inflation expected with 1000 samples.

    Under polygenicity, lambda_GC - 1 grows linearly with sample size, so the
    raw lambda is not comparable across studies. lambda_1000 removes that
    dependence (de Bakker et al. 2008; Freedman et al. 2004):

    * quantitative trait:  1 + (lambda - 1) * 1000 / N
    * case-control:        1 + (lambda - 1) * (1/n_cases + 1/n_controls) * 500
      (i.e. scaled to a study of 1000 cases and 1000 controls)
    """
    if lambda_gc is None or not np.isfinite(lambda_gc):
        return float("nan")
    if n_cases is not None and n_controls is not None:
        if n_cases <= 0 or n_controls <= 0:
            return float("nan")
        return float(1.0 + (lambda_gc - 1.0) * (1.0 / n_cases + 1.0 / n_controls) * 500.0)
    if not n_samples or n_samples <= 0:
        return float("nan")
    return float(1.0 + (lambda_gc - 1.0) * 1000.0 / n_samples)


def case_control_counts(phenotype: np.ndarray):
    """Return (n_cases, n_controls) for a binary 0/1 (or 1/2) phenotype, else (None, None).

    The larger code is treated as the case label.
    """
    y = np.asarray(phenotype, dtype=float)
    y = y[np.isfinite(y)]
    values = np.unique(y)
    if values.size != 2 or not (set(values.tolist()) <= {0.0, 1.0, 2.0}):
        return None, None
    n_cases = int(np.sum(y == values[1]))
    return n_cases, int(y.size - n_cases)
