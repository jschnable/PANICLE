"""
Effective number of independent marker estimation following the harmonised GEC
specification.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _NUMBA_AVAILABLE = False

from .data_types import GenotypeMap, GenotypeMatrix


class LDSparseMatrixProtocol:
    """Protocol-like base class describing the minimal LD interface we consume."""

    def get_all_unique_indexes(self) -> Sequence[int]:
        raise NotImplementedError

    def sub_dense_matrix(self, index_list: Sequence[int]) -> np.ndarray:
        raise NotImplementedError

    def release_ld_data(self) -> None:
        return None

    def convert_to_pvalue_coefficients(self, coeffs: Sequence[float]) -> None:
        raise NotImplementedError


@dataclass
class BlockStat:
    chrom: str
    start_pos: int
    end_pos: int
    n_snps: int
    Me: float
    largest_gap_bp: int


@dataclass
class ChromosomeResult:
    Me: float
    n_snps: int
    block_stats: List[BlockStat]
    missing_ld_snps: int = 0
    monomorphic_dropped: int = 0


DEFAULT_POLYNOMIAL_COEFFS: Tuple[float, ...] = (
    0.7723,
    -1.5659,
    1.2010,
    -0.2355,
    0.2184,
    0.6086,
    0.0,
)

_TRIU_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

def _get_triu_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    cached = _TRIU_CACHE.get(n)
    if cached is None:
        cached = np.triu_indices(n)
        _TRIU_CACHE[n] = cached
    return cached

if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _numba_standardise_corr(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples, n_markers = matrix.shape
        means = np.zeros(n_markers, dtype=np.float64)
        for j in range(n_markers):
            s = 0.0
            for i in range(n_samples):
                s += matrix[i, j]
            means[j] = s / n_samples if n_samples > 0 else 0.0
        for j in range(n_markers):
            m = means[j]
            for i in range(n_samples):
                matrix[i, j] -= m
        denom = n_samples - 1
        if denom < 1:
            denom = 1
        variances = np.zeros(n_markers, dtype=np.float64)
        for j in range(n_markers):
            s = 0.0
            for i in range(n_samples):
                val = matrix[i, j]
                s += val * val
            variances[j] = s / denom
        valid = np.zeros(n_markers, dtype=np.bool_)
        for j in range(n_markers):
            if variances[j] > 1e-12:
                inv_std = 1.0 / np.sqrt(variances[j])
                valid[j] = True
                for i in range(n_samples):
                    matrix[i, j] *= inv_std
            else:
                for i in range(n_samples):
                    matrix[i, j] = 0.0
        corr = np.zeros((n_markers, n_markers), dtype=np.float64)
        for a in range(n_markers):
            for b in range(a, n_markers):
                if valid[a] and valid[b]:
                    s = 0.0
                    for i in range(n_samples):
                        s += matrix[i, a] * matrix[i, b]
                    value = s / denom
                elif a == b:
                    value = 1.0
                else:
                    value = 0.0
                if a == b:
                    value = 1.0
                corr[a, b] = value
                corr[b, a] = value
        return corr, variances
else:

    def _numba_standardise_corr(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise RuntimeError("Numba is not available")


def _apply_polynomial_transform(matrix: np.ndarray, coeffs: Sequence[float]) -> None:
    """In-place polynomial transform mirroring convert2PValueCoefficient."""
    if matrix.size == 0:
        return

    # Work on upper triangle including diagonal.
    triu_idx = _get_triu_indices(matrix.shape[0])
    values = matrix[triu_idx]
    x = values * values
    poly_coeffs = coeffs[::-1]
    result = np.polynomial.polynomial.polyval(x, poly_coeffs)
    result = np.clip(result, 0.0, 1.0, out=result)

    matrix[triu_idx] = result
    matrix[(triu_idx[1], triu_idx[0])] = result  # mirror to lower triangle
    np.fill_diagonal(matrix, 1.0)


class _GenotypeLDSource(LDSparseMatrixProtocol):
    """Adapter exposing a genotype matrix chunk as an LD source."""

    def __init__(
        self,
        genotype: GenotypeMatrix,
        indices: Sequence[int],
        positions: Sequence[int],
        *,
        monomorphic_dropped: int = 0,
        prefiltered_low_variance: bool = False,
        preload_columns: bool = False,
        preload_max_elements: int = 64_000_000,
    ) -> None:
        if len(indices) != len(positions):
            raise ValueError("Indices and positions must be the same length")

        self._genotype = genotype
        self._indices = np.asarray(indices, dtype=int)
        self._positions = np.asarray(positions, dtype=int)
        order = np.argsort(self._positions, kind="mergesort")
        self._indices = self._indices[order]
        self._positions = self._positions[order]
        self._position_lookup: Dict[int, int] = {
            int(idx): int(pos) for idx, pos in zip(self._indices, self._positions)
        }
        self._index_column_lookup: Dict[int, int] = {
            int(idx): offset for offset, idx in enumerate(self._indices)
        }
        self.monomorphic_dropped = int(monomorphic_dropped)
        self.pre_filtered_low_variance = bool(prefiltered_low_variance)
        self._column_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._variance_cache: Dict[int, float] = {}
        self._stats_cache: Dict[int, Tuple[float, float]] = {}
        self._mean_cache: Dict[int, float] = {}
        self._cache_capacity = max(256, min(4096, len(self._indices)))
        self._low_variance_skipped = 0
        self._sample_count = int(self._genotype.n_individuals)
        self._preload_columns = bool(preload_columns)
        self._preload_max_elements = max(1, int(preload_max_elements))
        self._preloaded_columns: Optional[np.ndarray] = None
        self._preloaded_variances: Optional[np.ndarray] = None

    def _ensure_preloaded_columns(self) -> None:
        if not self._preload_columns:
            return
        if self._preloaded_columns is not None:
            return
        n_markers = int(self._indices.size)
        if n_markers == 0:
            return
        if (self._sample_count * n_markers) > self._preload_max_elements:
            # Disable preloading for very large chromosomes to avoid excessive memory use.
            self._preload_columns = False
            return

        columns = self._genotype.get_columns_imputed(self._indices, dtype=np.float64)
        if columns.ndim == 1:
            columns = columns.reshape(self._sample_count, -1)
        self._preloaded_columns = np.array(columns, dtype=np.float64, copy=True, order="F")
        if self._sample_count <= 1:
            self._preloaded_variances = np.zeros(n_markers, dtype=np.float64)
        else:
            denom = float(self._sample_count - 1)
            centered = self._preloaded_columns - np.mean(
                self._preloaded_columns,
                axis=0,
                dtype=np.float64,
            )
            variances = np.einsum(
                "ij,ij->j",
                centered,
                centered,
                dtype=np.float64,
            ) / denom
            tiny_neg = (variances < 0.0) & (np.abs(variances) < 1e-12)
            variances[tiny_neg] = 0.0
            self._preloaded_variances = variances

    def _enforce_cache_limit(self) -> None:
        while len(self._column_cache) > self._cache_capacity:
            self._column_cache.popitem(last=False)

    def _prefetch_range(self, start: int, end: int) -> None:
        self._ensure_preloaded_columns()
        if self._preloaded_columns is not None:
            return
        if end < start:
            return
        span = end - start + 1
        if span <= 0:
            return
        if span > self._cache_capacity * 4:
            return
        missing = [idx for idx in range(start, end + 1) if idx not in self._column_cache]
        if not missing:
            return
        fetched = self._genotype.get_columns_imputed(missing)
        if fetched.ndim == 1:
            fetched = fetched.reshape(self._genotype.n_individuals, -1)
        for offset, idx in enumerate(missing):
            col = np.ascontiguousarray(fetched[:, offset], dtype=np.float64)
            col.setflags(write=False)
            self._column_cache[idx] = col
        self._enforce_cache_limit()

    def _ensure_columns(self, indices: Sequence[int]) -> List[np.ndarray]:
        idx_list = [int(i) for i in indices]
        if not idx_list:
            return []
        self._ensure_preloaded_columns()
        if self._preloaded_columns is not None:
            return [self._preloaded_columns[:, self._index_column_lookup[idx]] for idx in idx_list]
        if len(idx_list) > 1:
            ordered = sorted(set(idx_list))
            span = ordered[-1] - ordered[0] + 1
            if span > len(ordered):
                self._prefetch_range(ordered[0], ordered[-1])
        missing = [idx for idx in idx_list if idx not in self._column_cache]
        if missing:
            fetched = self._genotype.get_columns_imputed(missing)
            if fetched.ndim == 1:
                fetched = fetched.reshape(self._genotype.n_individuals, -1)
            for offset, idx in enumerate(missing):
                col = np.ascontiguousarray(fetched[:, offset], dtype=np.float64)
                col.setflags(write=False)
                self._column_cache[idx] = col
            self._enforce_cache_limit()
        for idx in idx_list:
            try:
                self._column_cache.move_to_end(idx)
            except KeyError:
                # index may have been pruned from cache due to size; refetch
                fetched = self._genotype.get_columns_imputed([idx])
                col = np.ascontiguousarray(fetched[:, 0], dtype=np.float64)
                col.setflags(write=False)
                self._column_cache[idx] = col
                self._enforce_cache_limit()
        return [self._column_cache[idx] for idx in idx_list]

    def _get_columns_matrix(self, indices: Sequence[int]) -> np.ndarray:
        idx_list = [int(i) for i in indices]
        if not idx_list:
            raise ValueError("index_list must contain at least one marker")

        self._ensure_preloaded_columns()
        if self._preloaded_columns is not None:
            col_pos = np.asarray(
                [self._index_column_lookup[idx] for idx in idx_list],
                dtype=np.int64,
            )
            contiguous = col_pos.size <= 1 or np.all(np.diff(col_pos) == 1)
            if contiguous:
                view = self._preloaded_columns[:, int(col_pos[0]) : int(col_pos[-1]) + 1]
            else:
                view = self._preloaded_columns[:, col_pos]
            return np.array(view, dtype=np.float64, copy=True, order="F")

        cols = self._ensure_columns(idx_list)
        n_cols = len(cols)
        matrix = np.empty((self._sample_count, n_cols), dtype=np.float64, order="F")
        for offset, col in enumerate(cols):
            matrix[:, offset] = col
        return matrix

    def is_low_variance(self, index: int, *, threshold: float = 1e-12) -> bool:
        idx = int(index)
        if self._preloaded_variances is not None:
            col_pos = self._index_column_lookup.get(idx)
            if col_pos is not None:
                return float(self._preloaded_variances[col_pos]) <= threshold
        var = self._variance_cache.get(idx)
        if var is None:
            column = self._ensure_columns([idx])[0]
            if column.size <= 1:
                var = 0.0
            else:
                stats = self._get_column_stats(idx, column)
                var = self._variance_from_stats(stats)
            self._variance_cache[idx] = var
        return var <= threshold

    def record_low_variance_skip(self, count: int) -> None:
        if count:
            self._low_variance_skipped += int(count)

    @property
    def low_variance_skipped(self) -> int:
        return self._low_variance_skipped

    def _get_column_stats(
        self,
        index: int,
        column: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        idx = int(index)
        stats = self._stats_cache.get(idx)
        if stats is not None:
            return stats
        if column is None:
            column = self._ensure_columns([idx])[0]
        sum_val = float(np.sum(column, dtype=np.float64))
        sum_sq = float(np.dot(column, column))
        stats = (sum_val, sum_sq)
        self._stats_cache[idx] = stats
        self._mean_cache[idx] = sum_val / max(self._sample_count, 1)
        return stats

    def _variance_from_stats(self, stats: Tuple[float, float]) -> float:
        sum_val, sum_sq = stats
        n = self._sample_count
        if n <= 1:
            return 0.0
        denom = n - 1
        mean = sum_val / n
        var = (sum_sq - mean * mean * n) / denom
        if var < 0.0 and abs(var) < 1e-12:
            var = 0.0
        return float(var)

    def get_all_unique_indexes(self) -> Sequence[int]:
        return self._indices

    def get_position(self, index: int) -> int:
        return self._position_lookup[int(index)]

    def sub_dense_matrix(self, index_list: Sequence[int]) -> np.ndarray:
        if not index_list:
            raise ValueError("index_list must contain at least one marker")

        columns = self._get_columns_matrix(index_list)
        n_samples, n_markers = columns.shape
        if n_markers == 1 or n_samples <= 1:
            return np.ones((n_markers, n_markers), dtype=np.float64)

        use_numba = _NUMBA_AVAILABLE and n_markers > 1 and n_samples > 1
        if use_numba:
            try:
                corr, variances = _numba_standardise_corr(columns.copy())
            except Exception:  # pragma: no cover - fallback safety
                use_numba = False
        if not use_numba:
            means = np.mean(columns, axis=0, dtype=np.float64)
            columns -= means

            denom = max(n_samples - 1, 1)
            sumsq = np.einsum("ij,ij->j", columns, columns, dtype=np.float64)
            variances = sumsq / denom
            nonzero = variances > 1e-12

            if not np.any(nonzero):
                return np.identity(n_markers, dtype=np.float64)

            std = np.zeros_like(variances)
            std[nonzero] = np.sqrt(variances[nonzero])
            columns[:, nonzero] /= std[nonzero]
            columns[:, ~nonzero] = 0.0

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                corr = columns.T @ columns
            corr /= max(n_samples - 1, 1)
            np.nan_to_num(corr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            nonzero = variances > 1e-12
            corr = corr

        if (~nonzero).any():
            zero_idx = np.where(~nonzero)[0]
            corr[zero_idx, :] = 0.0
            corr[:, zero_idx] = 0.0
            corr[zero_idx, zero_idx] = 1.0

        np.fill_diagonal(corr, 1.0)
        return corr

    def release_ld_data(self) -> None:
        self._preloaded_columns = None
        self._preloaded_variances = None
        self._column_cache.clear()
        self._variance_cache.clear()
        self._stats_cache.clear()
        self._mean_cache.clear()
        return None

    def pair_correlation(self, idx_a: int, idx_b: int) -> Optional[float]:
        if idx_a == idx_b:
            return 1.0
        vectors = self._ensure_columns([idx_a, idx_b])
        if len(vectors) != 2:
            return None
        a, b = vectors
        n = self._sample_count
        if n <= 1:
            return None
        stats_a = self._get_column_stats(idx_a, a)
        stats_b = self._get_column_stats(idx_b, b)
        var_a = self._variance_from_stats(stats_a)
        var_b = self._variance_from_stats(stats_b)
        if var_a <= 1e-12 or var_b <= 1e-12:
            return None
        sum_a, _ = stats_a
        sum_b, _ = stats_b
        sum_ab = float(np.dot(a, b))
        mean_a = sum_a / n
        mean_b = sum_b / n
        cov = sum_ab - n * mean_a * mean_b
        denom = (n - 1) * np.sqrt(var_a) * np.sqrt(var_b)
        if denom <= 0.0:
            return None
        corr = cov / denom
        if np.isnan(corr):
            return None
        return corr

    def adjacent_correlations(self, indices: Sequence[int]) -> np.ndarray:
        idx_list = [int(i) for i in indices]
        if len(idx_list) <= 1:
            return np.empty(0, dtype=np.float64)
        if self._sample_count <= 1:
            return np.full(len(idx_list) - 1, np.nan, dtype=np.float64)
        if (self._sample_count * len(idx_list)) > self._preload_max_elements:
            # Avoid allocating very large temporary matrices for adjacency precompute.
            return np.full(len(idx_list) - 1, np.nan, dtype=np.float64)

        columns = self._get_columns_matrix(idx_list)
        n_samples = columns.shape[0]
        if n_samples <= 1:
            return np.full(len(idx_list) - 1, np.nan, dtype=np.float64)

        columns -= np.mean(columns, axis=0, dtype=np.float64)
        denom = max(n_samples - 1, 1)
        sumsq = np.einsum("ij,ij->j", columns, columns, dtype=np.float64)
        variances = sumsq / denom
        valid = variances > 1e-12
        std = np.zeros_like(variances)
        std[valid] = np.sqrt(variances[valid])

        cross = np.einsum("ij,ij->j", columns[:, :-1], columns[:, 1:], dtype=np.float64)
        den = denom * std[:-1] * std[1:]
        corr = np.full(len(idx_list) - 1, np.nan, dtype=np.float64)
        ok = valid[:-1] & valid[1:] & (den > 0.0)
        corr[ok] = cross[ok] / den[ok]
        finite = np.isfinite(corr)
        corr[finite] = np.clip(corr[finite], -1.0, 1.0)
        return corr


def _normalise_ld_sources(
    ld_sources: Union[
        LDSparseMatrixProtocol,
        Mapping[str, LDSparseMatrixProtocol],
        Iterable[Tuple[str, LDSparseMatrixProtocol]],
    ],
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, LDSparseMatrixProtocol]:
    if isinstance(ld_sources, LDSparseMatrixProtocol):
        chrom = "unknown"
        if metadata is not None:
            chrom = str(metadata.get("chromosome", chrom))
        return {chrom: ld_sources}

    if isinstance(ld_sources, Mapping):
        return dict(ld_sources)

    normalised: Dict[str, LDSparseMatrixProtocol] = {}
    for chrom, source in ld_sources:
        normalised[str(chrom)] = source
    return normalised


def _construct_blocks(
    chrom: str,
    ld_source: LDSparseMatrixProtocol,
    *,
    max_window_bp: int,
    corr_cutoff: float,
    gap_snp_limit: int,
    span_bp_limit: int,
) -> Tuple[List[List[int]], List[BlockStat], int]:
    indices = list(ld_source.get_all_unique_indexes())
    if not indices:
        return [], [], 0

    if hasattr(ld_source, "get_position"):
        positions = [int(getattr(ld_source, "get_position")(idx)) for idx in indices]
    else:
        positions = list(range(len(indices)))

    raw_entries = sorted(zip(indices, positions), key=lambda x: x[1])
    total_indices = len(raw_entries)

    entries = raw_entries
    low_var_checker = getattr(ld_source, "is_low_variance", None)
    already_prefiltered = bool(getattr(ld_source, "pre_filtered_low_variance", False))
    if callable(low_var_checker) and not already_prefiltered:
        filtered: List[Tuple[int, int]] = []
        skipped = 0
        for idx, pos in raw_entries:
            if low_var_checker(idx):
                skipped += 1
                continue
            filtered.append((idx, pos))
        if skipped and hasattr(ld_source, "record_low_variance_skip"):
            getattr(ld_source, "record_low_variance_skip")(skipped)
        entries = filtered

    blocks: List[List[int]] = []
    stats: List[BlockStat] = []

    if not entries:
        return [], [], total_indices

    ordered_indices = [idx for idx, _ in entries]
    adjacent_corr_values: Optional[np.ndarray] = None
    adjacent_corr_fn = getattr(ld_source, "adjacent_correlations", None)
    if callable(adjacent_corr_fn):
        try:
            adjacent_corr_values = np.asarray(adjacent_corr_fn(ordered_indices), dtype=np.float64)
            if adjacent_corr_values.ndim != 1 or adjacent_corr_values.size != max(len(ordered_indices) - 1, 0):
                adjacent_corr_values = None
        except Exception:
            adjacent_corr_values = None

    n = len(entries)
    start = 0
    while start < n:
        block_indices = [entries[start][0]]
        block_positions = [entries[start][1]]
        largest_gap = 0
        gap_run = 0
        end = start + 1

        while end < n:
            curr_idx, curr_pos = entries[end]
            prev_idx, prev_pos = entries[end - 1]

            if curr_pos - entries[start][1] > span_bp_limit:
                break
            if curr_pos - prev_pos > max_window_bp:
                break

            pair_corr = None
            pair_idx = end - 1
            if (
                adjacent_corr_values is not None
                and 0 <= pair_idx < adjacent_corr_values.size
            ):
                pair_value = adjacent_corr_values[pair_idx]
                if np.isfinite(pair_value):
                    pair_corr = float(pair_value)
            if pair_corr is None and hasattr(ld_source, "pair_correlation"):
                pair_corr = getattr(ld_source, "pair_correlation")(prev_idx, curr_idx)

            if pair_corr is None:
                gap_run += 1
                if gap_run > gap_snp_limit:
                    break
            else:
                gap_run = 0
                if abs(pair_corr) < corr_cutoff:
                    break

            largest_gap = max(largest_gap, curr_pos - prev_pos)
            block_indices.append(curr_idx)
            block_positions.append(curr_pos)
            end += 1

        blocks.append(block_indices)
        stats.append(
            BlockStat(
                chrom=chrom,
                start_pos=block_positions[0],
                end_pos=block_positions[-1],
                n_snps=len(block_indices),
                Me=0.0,
                largest_gap_bp=largest_gap,
            )
        )
        start = max(end, start + 1)

    return blocks, stats, total_indices


def _prune_redundant_snps(
    matrix: np.ndarray,
    indices: List[int],
    threshold: Optional[float],
) -> Tuple[np.ndarray, List[int]]:
    if threshold is None or matrix.shape[0] <= 1:
        return matrix, indices

    keep_mask = np.ones(matrix.shape[0], dtype=bool)
    for i in range(matrix.shape[0]):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, matrix.shape[0]):
            if not keep_mask[j]:
                continue
            if abs(matrix[i, j]) >= threshold:
                keep_mask[j] = False

    if keep_mask.all():
        return matrix, indices

    retained_indices = [idx for idx, keep in zip(indices, keep_mask) if keep]
    pruned_matrix = matrix[np.ix_(keep_mask, keep_mask)]
    return pruned_matrix, retained_indices


def _process_chromosome(
    chrom: str,
    ld_source: LDSparseMatrixProtocol,
    *,
    max_window_bp: int,
    corr_cutoff: float,
    gap_snp_limit: int,
    span_bp_limit: int,
    prune_redundant_threshold: Optional[float],
    polynomial_coeffs: Sequence[float],
) -> ChromosomeResult:
    converted = False
    convert_method = getattr(ld_source, "convert_to_pvalue_coefficients", None)
    if convert_method is not None:
        try:
            convert_method(polynomial_coeffs)
            converted = True
        except NotImplementedError:
            converted = False
        except Exception:
            converted = False

    blocks, stats, total_indices = _construct_blocks(
        chrom,
        ld_source,
        max_window_bp=max_window_bp,
        corr_cutoff=corr_cutoff,
        gap_snp_limit=gap_snp_limit,
        span_bp_limit=span_bp_limit,
    )

    Me_total = 0.0

    for block_indices, block_stat in zip(blocks, stats):
        ld_matrix = ld_source.sub_dense_matrix(block_indices)

        if ld_matrix.size == 0 or ld_matrix.shape[0] != ld_matrix.shape[1]:
            raise ValueError("LD matrix must be non-empty and square")

        if not converted:
            _apply_polynomial_transform(ld_matrix, polynomial_coeffs)

        ld_matrix, block_indices = _prune_redundant_snps(
            ld_matrix, list(block_indices), prune_redundant_threshold
        )

        ld_matrix = (ld_matrix + ld_matrix.T) / 2.0
        np.clip(ld_matrix, 0.0, 1.0, out=ld_matrix)
        np.fill_diagonal(ld_matrix, 1.0)

        try:
            eigenvalues = np.linalg.eigvalsh(ld_matrix)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"Eigen decomposition failed for {chrom}") from exc

        capped = np.maximum(eigenvalues - 1.0, 0.0)
        Me_block = ld_matrix.shape[0] - float(np.sum(capped))
        if Me_block < 0:
            Me_block = 0.0

        Me_total += Me_block
        block_stat.n_snps = len(block_indices)
        block_stat.Me = Me_block

    covered = sum(len(block) for block in blocks)
    missing_ld_snps = max(total_indices - covered, 0)
    Me_total += float(missing_ld_snps)
    n_snps = total_indices

    monomorphic_dropped = getattr(ld_source, "monomorphic_dropped", 0)
    extra_low_var = getattr(ld_source, "low_variance_skipped", 0)
    monomorphic_dropped += extra_low_var

    result = ChromosomeResult(
        Me=Me_total,
        n_snps=n_snps,
        block_stats=stats,
        missing_ld_snps=missing_ld_snps,
        monomorphic_dropped=monomorphic_dropped,
    )

    ld_source.release_ld_data()

    return result


def estimate_effective_tests(
    ld_sources: Union[
        LDSparseMatrixProtocol,
        Mapping[str, LDSparseMatrixProtocol],
        Iterable[Tuple[str, LDSparseMatrixProtocol]],
    ],
    *,
    max_window_bp: int = 3_000_000,
    corr_cutoff: float = 0.7,
    gap_snp_limit: int = 500,
    span_bp_limit: int = 3_000_000,
    prune_redundant_threshold: Optional[float] = 0.9988,
    polynomial_coeffs: Optional[Sequence[float]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    executor: Optional[Executor] = None,
) -> Dict[str, Any]:
    polynomial_coeffs = (
        tuple(polynomial_coeffs) if polynomial_coeffs is not None else DEFAULT_POLYNOMIAL_COEFFS
    )

    sources = _normalise_ld_sources(ld_sources, metadata)
    if not sources:
        return {
            "Me": 0.0,
            "per_chromosome": {},
            "block_stats": [],
            "total_snps": 0,
            "parameters": {
                "max_window_bp": max_window_bp,
                "corr_cutoff": corr_cutoff,
                "gap_snp_limit": gap_snp_limit,
                "span_bp_limit": span_bp_limit,
                "prune_redundant_threshold": prune_redundant_threshold,
                "polynomial_coeffs": polynomial_coeffs,
            },
            "metadata": dict(metadata) if metadata else {},
        }

    per_chromosome: MutableMapping[str, ChromosomeResult] = {}

    if executor is not None:
        futures: Dict[Future, str] = {}
        for chrom, source in sources.items():
            future = executor.submit(
                _process_chromosome,
                chrom,
                source,
                max_window_bp=max_window_bp,
                corr_cutoff=corr_cutoff,
                gap_snp_limit=gap_snp_limit,
                span_bp_limit=span_bp_limit,
                prune_redundant_threshold=prune_redundant_threshold,
                polynomial_coeffs=polynomial_coeffs,
            )
            futures[future] = chrom

        for future, chrom in futures.items():
            per_chromosome[chrom] = future.result()
    else:
        for chrom, source in sources.items():
            per_chromosome[chrom] = _process_chromosome(
                chrom,
                source,
                max_window_bp=max_window_bp,
                corr_cutoff=corr_cutoff,
                gap_snp_limit=gap_snp_limit,
                span_bp_limit=span_bp_limit,
                prune_redundant_threshold=prune_redundant_threshold,
                polynomial_coeffs=polynomial_coeffs,
            )

    total_me = sum(result.Me for result in per_chromosome.values())
    rounded_total_me = int(round(total_me))
    block_stats = [
        {
            "chrom": block.chrom,
            "start_pos": block.start_pos,
            "end_pos": block.end_pos,
            "n_snps": block.n_snps,
            "Me": block.Me,
            "largest_gap_bp": block.largest_gap_bp,
        }
        for result in per_chromosome.values()
        for block in result.block_stats
    ]
    total_snps = sum(result.n_snps + result.monomorphic_dropped for result in per_chromosome.values())
    total_monomorphic = sum(result.monomorphic_dropped for result in per_chromosome.values())

    return {
        "Me": rounded_total_me,
        "per_chromosome": {
            chrom: {
                "Me": int(round(result.Me)),
                "n_snps": result.n_snps,
                "block_stats": [
                    {
                        "chrom": block.chrom,
                        "start_pos": block.start_pos,
                        "end_pos": block.end_pos,
                        "n_snps": block.n_snps,
                        "Me": block.Me,
                        "largest_gap_bp": block.largest_gap_bp,
                    }
                    for block in result.block_stats
                ],
                "missing_ld_snps": result.missing_ld_snps,
                "monomorphic_dropped": result.monomorphic_dropped,
            }
            for chrom, result in per_chromosome.items()
        },
        "block_stats": block_stats,
        "total_snps": total_snps,
        "dropped_monomorphic_total": total_monomorphic,
        "parameters": {
            "max_window_bp": max_window_bp,
            "corr_cutoff": corr_cutoff,
            "gap_snp_limit": gap_snp_limit,
            "span_bp_limit": span_bp_limit,
            "prune_redundant_threshold": prune_redundant_threshold,
            "polynomial_coeffs": polynomial_coeffs,
        },
        "metadata": dict(metadata) if metadata else {},
    }


def make_ld_sources_from_genotype(
    genotype: GenotypeMatrix,
    genotype_map: GenotypeMap,
) -> Dict[str, LDSparseMatrixProtocol]:
    """Construct LD sources per chromosome from a genotype matrix + map."""
    map_df = genotype_map.to_dataframe()
    if "CHROM" not in map_df.columns or "POS" not in map_df.columns:
        raise ValueError("GenotypeMap must contain CHROM and POS columns")

    per_chrom_indices: Dict[str, List[int]] = {}
    per_chrom_positions: Dict[str, List[int]] = {}
    for idx, (chrom, pos) in enumerate(zip(map_df["CHROM"], map_df["POS"])):
        chrom_str = str(chrom)
        per_chrom_indices.setdefault(chrom_str, []).append(idx)
        per_chrom_positions.setdefault(chrom_str, []).append(int(pos))

    ld_sources: Dict[str, LDSparseMatrixProtocol] = {}
    for chrom, indices in per_chrom_indices.items():
        positions = per_chrom_positions[chrom]
        entries = list(zip(indices, positions))
        filtered_entries: List[Tuple[int, int]] = []
        dropped = 0

        if entries:
            chunk_size = 1024
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start : start + chunk_size]
                chunk_indices = [idx for idx, _ in chunk]
                chunk_positions = [pos for _, pos in chunk]
                columns = genotype.get_columns_imputed(chunk_indices)
                variances = np.var(columns, axis=0)
                keep_mask = np.asarray(variances > 1e-12)
                if keep_mask.ndim == 0:
                    keep_mask = np.asarray([bool(keep_mask)])
                for (entry, keep) in zip(chunk, keep_mask.tolist()):
                    if keep:
                        filtered_entries.append(entry)
                    else:
                        dropped += 1
                if len(chunk) > len(keep_mask):
                    dropped += len(chunk) - len(keep_mask)

        if filtered_entries:
            filtered_indices = [int(idx) for idx, _ in filtered_entries]
            filtered_positions = [int(pos) for _, pos in filtered_entries]
        else:
            filtered_indices = []
            filtered_positions = []

        ld_sources[chrom] = _GenotypeLDSource(
            genotype,
            filtered_indices,
            filtered_positions,
            monomorphic_dropped=dropped,
            prefiltered_low_variance=True,
            preload_columns=True,
        )

    return ld_sources


def estimate_effective_tests_from_genotype(
    genotype: GenotypeMatrix,
    genotype_map: GenotypeMap,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience wrapper combining LD construction and Me estimation.

    Optional keyword arguments:
      - ``ncpus`` or ``cpu``: number of workers for per-chromosome processing.
        ``0`` means "use all available CPUs". ``1`` (default) runs serially.
      - ``executor``: pre-built ``concurrent.futures.Executor``. If provided,
        it takes precedence over ``ncpus``/``cpu``.
      - Any other kwargs are passed through to ``estimate_effective_tests``.
    """
    ld_sources = make_ld_sources_from_genotype(genotype, genotype_map)
    metadata = kwargs.pop("metadata", None)
    executor = kwargs.get("executor")

    if executor is None:
        raw_workers = kwargs.pop("ncpus", kwargs.pop("cpu", None))
        if raw_workers is not None:
            try:
                workers = int(raw_workers)
            except (TypeError, ValueError):
                raise ValueError("ncpus/cpu for effective tests must be an integer")
            if workers < 0:
                raise ValueError("ncpus/cpu for effective tests must be >= 0")
            if workers == 0:
                workers = max(1, os.cpu_count() or 1)
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    return estimate_effective_tests(
                        ld_sources,
                        metadata=metadata,
                        executor=pool,
                        **kwargs,
                    )

    return estimate_effective_tests(
        ld_sources,
        metadata=metadata,
        **kwargs,
    )
