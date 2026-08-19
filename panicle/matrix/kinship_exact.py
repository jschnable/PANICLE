"""Exact VanRaden Gram for complete 0/1/2 int8 dosages.

Uncentered ``ZZᵀ`` is an integer matrix. A batch of 0/1/2 in float32 with
width ``w`` has Gram entries ``≤ 4w``. For the default ``w = 5000`` that
is 20,000, exactly representable in float32, so the per-batch GEMM does
not depend on how OpenBLAS partitions the product. Batches are summed in
float64 (total ``≤ 4 * n_markers``, far below ``2⁵³``). Centering is the
algebraic identity

    G = ZZᵀ − 1 sᵀ − s 1ᵀ + (μ·μ) 11ᵀ

with ``s = Zμ``, all in float64. VanRaden scaling is a scalar divide.

The resulting ``K`` is a property of the dosages, not of BLAS build,
thread count, or batch width. It will not match the previous
float32-center-then-GEMM path; that is a deliberate results change.

Missing / non-{0,1,2} dosages stay on the older float32 path.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..utils.data_types import GenotypeMatrix

# 4 * 1e6 = 4e6 ≪ 2²⁴, so a float32 batch Gram of 0/1/2 stays exact.
EXACT_BATCH_MAX = 1_000_000


def can_use_exact_gram(genotype: Union[GenotypeMatrix, np.ndarray]) -> bool:
    """True when dosages are complete int8 (imputed matrix or 0/1/2 array)."""
    if isinstance(genotype, GenotypeMatrix):
        if not bool(genotype.is_imputed):
            return False
        if bool(getattr(genotype, "is_transposed", False)):
            return False
        storage = getattr(genotype, "_storage", None)
        return isinstance(storage, np.ndarray) and storage.dtype == np.int8
    if not isinstance(genotype, np.ndarray) or genotype.ndim != 2:
        return False
    if genotype.dtype != np.int8:
        return False
    if genotype.size == 0:
        return True
    return int(genotype.min()) >= 0 and int(genotype.max()) <= 2


class UncenteredGram:
    """Running ``ZZᵀ`` plus ``‖col_sums‖²`` for algebraic centering.

    ``t = Z · col_sums`` is the row-sum of the integer Gram, so it is
    recovered once as ``S.sum(axis=1)``. ``ss`` is accumulated from
    per-batch column sums of the float32 dosages (exact for 0/1/2).
    """

    def __init__(self, n_individuals: int):
        n = int(n_individuals)
        self.n_individuals = n
        self.S = np.zeros((n, n), dtype=np.float64)
        self.ss = np.int64(0)

    def add_batch(self, Z_int8: np.ndarray) -> None:
        if Z_int8.ndim != 2:
            raise ValueError("Z_int8 must be 2D")
        if Z_int8.shape[0] != self.n_individuals:
            raise ValueError("batch row count must match n_individuals")
        if Z_int8.shape[1] == 0 or self.n_individuals == 0:
            return
        Z = np.ascontiguousarray(Z_int8, dtype=np.float32)
        # Float32 Z@Z.T is an exact integer on 0/1/2 at this width; add
        # it straight into the float64 accumulator. ssyrk is slower than
        # GEMM for this (n=730, k=5k) shape on OpenBLAS.
        self.S += Z @ Z.T
        col_sums = Z.sum(axis=0, dtype=np.float64)
        self.ss += np.int64(col_sums @ col_sums)

    def add_gram(self, other: "UncenteredGram") -> None:
        if other.n_individuals != self.n_individuals:
            raise ValueError("cannot add Grams with different n_individuals")
        self.S += other.S
        self.ss += other.ss

    def centered(self) -> np.ndarray:
        """Return the column-centered Gram ``(Z − 1μᵀ)(Z − 1μᵀ)ᵀ``."""
        n = float(self.n_individuals)
        if n <= 0:
            return self.S.copy()
        t = self.S.sum(axis=1)
        s = t / n
        sigma = float(self.ss) / (n * n)
        G = self.S - s[:, np.newaxis] - s[np.newaxis, :] + sigma
        return (G + G.T) * 0.5


def int8_marker_block(
    genotype: Union[GenotypeMatrix, np.ndarray],
    start: int,
    end: int,
) -> np.ndarray:
    """Native int8 columns ``[start, end)`` in individual-major order."""
    if isinstance(genotype, GenotypeMatrix):
        return np.ascontiguousarray(genotype.get_batch(start, end))
    return np.ascontiguousarray(genotype[:, start:end])


def int8_marker_columns(
    genotype: Union[GenotypeMatrix, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    """Native int8 columns at ``indices`` in individual-major order."""
    idx = np.asarray(indices, dtype=np.int64)
    if isinstance(genotype, GenotypeMatrix):
        return np.ascontiguousarray(genotype.get_columns(idx))
    return np.ascontiguousarray(genotype[:, idx])


def accumulate_uncentered(
    genotype: Union[GenotypeMatrix, np.ndarray],
    n_individuals: int,
    n_markers: int,
    max_line: int,
    *,
    indices: Optional[np.ndarray] = None,
) -> UncenteredGram:
    """Stream ``max_line``-wide int8 batches into an :class:`UncenteredGram`."""
    gram = UncenteredGram(n_individuals)
    width = max(1, min(int(max_line), EXACT_BATCH_MAX))
    if indices is None:
        for start in range(0, int(n_markers), width):
            end = min(start + width, int(n_markers))
            gram.add_batch(int8_marker_block(genotype, start, end))
        return gram
    idx = np.asarray(indices, dtype=np.int64)
    for start in range(0, idx.size, width):
        gram.add_batch(int8_marker_columns(genotype, idx[start : start + width]))
    return gram


def vanraden_from_centered(G: np.ndarray) -> np.ndarray:
    """Scale a centered Gram by the mean diagonal (VanRaden)."""
    kin = np.asarray(G, dtype=np.float64)
    mean_diag = float(np.mean(np.diag(kin)))
    if mean_diag > 0.0:
        kin = kin / mean_diag
    return kin
