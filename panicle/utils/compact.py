"""Run-length column compaction for row-major integer matrices.

``A[:, keep]`` on a C-contiguous ``(n, ncols)`` int8 array is a per-element
fancy-index. When ``keep`` is strictly increasing with short holes, the same
copy is a list of contiguous runs and can be done with a sequential kernel.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba is a hard dependency
    _NUMBA_AVAILABLE = False
    njit = None
    prange = range


def runs_from_keep_indices(keep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a strictly increasing keep index into (start, length) runs.

    ``start[i]`` is a column index into the source matrix; ``length[i]`` is
    the number of consecutive source columns in that run. ``sum(length)``
    equals ``keep.size``.
    """
    keep = np.asarray(keep, dtype=np.int64)
    if keep.ndim != 1:
        raise ValueError("keep indices must be 1D")
    if keep.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    if keep.size == 1:
        return keep.copy(), np.ones(1, dtype=np.int64)

    gaps = np.diff(keep)
    boundaries = np.flatnonzero(gaps != 1) + 1
    n_runs = int(boundaries.size) + 1
    starts_at = np.empty(n_runs, dtype=np.int64)
    starts_at[0] = 0
    if boundaries.size:
        starts_at[1:] = boundaries
    ends_at = np.empty(n_runs, dtype=np.int64)
    if boundaries.size:
        ends_at[:-1] = boundaries
    ends_at[-1] = keep.size
    return keep[starts_at], ends_at - starts_at


def keep_indices_are_strictly_increasing(keep: np.ndarray) -> bool:
    """True when ``keep`` is a 1D strictly increasing integer index."""
    if keep.size <= 1:
        return True
    return bool(np.all(np.diff(keep) > 0))


if _NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True, nogil=True)
    def _compact_runs_int8(src, starts, lens, dst):
        n_rows = src.shape[0]
        n_runs = starts.shape[0]
        for i in prange(n_rows):
            w = 0
            for r in range(n_runs):
                s = starts[r]
                length = lens[r]
                for j in range(length):
                    dst[i, w + j] = src[i, s + j]
                w += length

    @njit(cache=True, parallel=True, nogil=True)
    def _compact_runs_int8_rows(src, row_idx, starts, lens, dst):
        n_rows = row_idx.shape[0]
        n_runs = starts.shape[0]
        for i in prange(n_rows):
            src_i = row_idx[i]
            w = 0
            for r in range(n_runs):
                s = starts[r]
                length = lens[r]
                for j in range(length):
                    dst[i, w + j] = src[src_i, s + j]
                w += length

    @njit(cache=True, nogil=True)
    def _column_sums_int8_rows(src, row_idx):
        n_rows = row_idx.shape[0]
        n_cols = src.shape[1]
        out = np.zeros(n_cols, dtype=np.float64)
        for i in range(n_rows):
            src_i = row_idx[i]
            for j in range(n_cols):
                out[j] += src[src_i, j]
        return out

    @njit(cache=True, parallel=True, nogil=True)
    def _int8_to_float32(src, dst):
        n_rows, n_cols = src.shape
        for i in prange(n_rows):
            for j in range(n_cols):
                dst[i, j] = np.float32(src[i, j])


def compact_columns_int8(
    src: np.ndarray,
    keep: np.ndarray,
    row_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Copy kept columns of a C-contiguous int8 matrix into a packed buffer.

    ``keep`` must be strictly increasing and in bounds. When ``row_idx`` is
    given, only those source rows are copied (in that order). The result is a
    new C-contiguous ``(n_out, keep.size)`` int8 array, bit-identical to
    ``np.ascontiguousarray(src[np.ix_(row_idx, keep)])``.
    """
    src = np.asarray(src)
    keep = np.asarray(keep, dtype=np.int64)
    if src.ndim != 2:
        raise ValueError("src must be 2D")
    if src.dtype != np.int8:
        raise TypeError("src must be int8")
    if keep.ndim != 1:
        raise ValueError("keep must be 1D")
    n_src_rows, n_cols = src.shape
    if keep.size and (int(keep[0]) < 0 or int(keep[-1]) >= n_cols):
        raise IndexError("keep indices are out of bounds")
    if not keep_indices_are_strictly_increasing(keep):
        raise ValueError("keep indices must be strictly increasing")

    if row_idx is None:
        n_out = n_src_rows
        row_idx_i64 = None
    else:
        row_idx_i64 = np.asarray(row_idx, dtype=np.int64)
        if row_idx_i64.ndim != 1:
            raise ValueError("row_idx must be 1D")
        if row_idx_i64.size and (
            int(row_idx_i64.min()) < 0 or int(row_idx_i64.max()) >= n_src_rows
        ):
            raise IndexError("row_idx is out of bounds")
        n_out = int(row_idx_i64.size)

    dst = np.empty((n_out, keep.size), dtype=np.int8)
    if keep.size == 0 or n_out == 0:
        return dst

    if not src.flags.c_contiguous:
        src = np.ascontiguousarray(src)

    starts, lens = runs_from_keep_indices(keep)
    if _NUMBA_AVAILABLE:
        if row_idx_i64 is None:
            _compact_runs_int8(src, starts, lens, dst)
        else:
            _compact_runs_int8_rows(src, row_idx_i64, starts, lens, dst)
    else:  # pragma: no cover
        src_rows = src if row_idx_i64 is None else src[row_idx_i64, :]
        write = 0
        for start, length in zip(starts.tolist(), lens.tolist()):
            dst[:, write : write + length] = src_rows[:, start : start + length]
            write += int(length)
    return dst


def column_sums_int8(
    src: np.ndarray,
    row_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-column sums of an int8 matrix, optionally over a row subset."""
    src = np.asarray(src)
    if src.ndim != 2 or src.dtype != np.int8:
        raise TypeError("src must be a 2D int8 array")
    if row_idx is None:
        return src.sum(axis=0, dtype=np.float64)
    row_idx_i64 = np.asarray(row_idx, dtype=np.int64)
    if row_idx_i64.size == 0:
        return np.zeros(src.shape[1], dtype=np.float64)
    if not src.flags.c_contiguous:
        src = np.ascontiguousarray(src)
    if _NUMBA_AVAILABLE:
        return _column_sums_int8_rows(src, row_idx_i64)
    return src[row_idx_i64, :].sum(axis=0, dtype=np.float64)


def int8_to_float32(src: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Bit-identical ``src.astype(np.float32)`` for a C-contiguous int8 matrix.

    Integer-to-float32 conversion is exact for every int8 value. The numba
    parallel kernel matches numpy's ``astype`` elementwise; it exists because
    numpy's cast is single-threaded.
    """
    src = np.asarray(src)
    if src.ndim != 2 or src.dtype != np.int8:
        raise TypeError("src must be a 2D int8 array")
    if not src.flags.c_contiguous:
        src = np.ascontiguousarray(src)
    if out is None:
        dst = np.empty(src.shape, dtype=np.float32)
    else:
        dst = np.asarray(out)
        if dst.shape != src.shape or dst.dtype != np.float32:
            raise ValueError("out must be float32 with the same shape as src")
        if not dst.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")
    if src.size == 0:
        return dst
    # Parallel launch loses on 5k-column kinship batches; numpy wins
    # there and the kernel only pays off on whole-group converts.
    if _NUMBA_AVAILABLE and src.size >= 8_000_000:
        _int8_to_float32(src, dst)
    else:
        np.copyto(dst, src, casting="unsafe")
    return dst


def can_compact_int8_columns(src: np.ndarray, keep: np.ndarray) -> bool:
    """Whether ``compact_columns_int8`` can replace a fancy-index gather."""
    if not isinstance(src, np.ndarray) or src.ndim != 2 or src.dtype != np.int8:
        return False
    if keep.ndim != 1 or keep.size == 0:
        return False
    return keep_indices_are_strictly_increasing(keep)
