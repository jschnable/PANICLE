"""In-sample linkage disequilibrium: LD pruning and pairwise r^2.

Many plant genomes have no external LD reference panel, and LD blocks can be
long (selfing, inversions, introgressions). Everything here is therefore
estimated from the analysed genotypes themselves.

Pruning walks each chromosome in position order and computes marker-marker
correlations in blocks (``Z_rows.T @ Z_cols / n`` on standardised genotypes),
so memory is bounded by ``n_individuals x (chunk + max_offset)`` regardless of
the number of markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Sequence, Tuple, Union

import numpy as np
from numba import njit

from ..utils.data_types import GenotypeMatrix

GenotypeLike = Union[GenotypeMatrix, np.ndarray]

DEFAULT_CHUNK = 2048


# ---------------------------------------------------------------------------
# Genotype access helpers
# ---------------------------------------------------------------------------

def _n_individuals(genotype: GenotypeLike) -> int:
    return genotype.n_individuals if isinstance(genotype, GenotypeMatrix) else genotype.shape[0]


def _raw_columns(genotype: GenotypeLike, indices: np.ndarray) -> np.ndarray:
    if isinstance(genotype, GenotypeMatrix):
        return genotype.get_columns(indices, dtype=np.float32, copy=True)
    return np.asarray(genotype[:, indices], dtype=np.float32).copy()


def standardized_columns(genotype: GenotypeLike, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean-imputed, unit-variance genotype columns and their MAF.

    Missing calls (``-9`` or NaN) are set to the marker mean, i.e. they
    contribute zero to every correlation. Monomorphic markers are returned as
    all-zero columns (r = 0 with everything).
    """
    X = _raw_columns(genotype, np.asarray(indices, dtype=np.int64))
    missing = (X == -9) | ~np.isfinite(X)
    X[missing] = np.nan
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    X = np.where(missing, mean[np.newaxis, :], X)
    X -= mean[np.newaxis, :]
    sd = np.sqrt(np.mean(X * X, axis=0))
    ok = sd > 1e-8
    X[:, ok] /= sd[ok][np.newaxis, :]
    X[:, ~ok] = 0.0
    p = mean / 2.0
    maf = np.minimum(p, 1.0 - p).astype(np.float64)
    maf[~ok] = 0.0
    return X, maf


def chromosome_order(chromosomes: Sequence, positions: Sequence) -> Dict[str, np.ndarray]:
    """Map each chromosome label to its marker indices sorted by position."""
    chrom = np.asarray(chromosomes).astype(str)
    pos = np.asarray(positions, dtype=np.float64)
    groups: Dict[str, np.ndarray] = {}
    for label in dict.fromkeys(chrom.tolist()):
        idx = np.flatnonzero(chrom == label)
        groups[label] = idx[np.argsort(pos[idx], kind="stable")]
    return groups


# ---------------------------------------------------------------------------
# Block iterator over forward (j > i) marker pairs within a window
# ---------------------------------------------------------------------------

@dataclass
class _RBlock:
    rows: np.ndarray        # local (within-chromosome) row indices
    cols: np.ndarray        # local column indices, cols[0] == rows[0]
    r: np.ndarray           # correlation block, shape (len(rows), len(cols))
    in_window: np.ndarray   # bool mask: j > i and within the distance/offset window


def _iter_forward_blocks(
    Z_loader,
    pos: np.ndarray,
    n_ind: int,
    window_bp: float,
    max_offset: int,
    chunk: int,
) -> Iterator[_RBlock]:
    """Yield correlation blocks covering every pair (i, j>i) inside the window.

    ``Z_loader(a, b)`` returns standardised columns for local markers [a, b).
    """
    m = pos.size
    for a in range(0, m, chunk):
        b = min(a + chunk, m)
        # furthest column any row in [a, b) can reach
        reach = int(np.searchsorted(pos, pos[b - 1] + window_bp, side="right"))
        e = min(m, reach, b - 1 + max_offset + 1)
        Z = Z_loader(a, e)
        r = (Z[:, : b - a].T @ Z) / np.float32(n_ind)
        rows = np.arange(a, b)
        cols = np.arange(a, e)
        offset = cols[np.newaxis, :] - rows[:, np.newaxis]
        dist = pos[cols][np.newaxis, :] - pos[rows][:, np.newaxis]
        in_window = (offset > 0) & (offset <= max_offset) & (dist <= window_bp)
        yield _RBlock(rows, cols, r, in_window)


def _make_loader(genotype: GenotypeLike, global_idx: np.ndarray):
    def load(a: int, b: int) -> np.ndarray:
        Z, _ = standardized_columns(genotype, global_idx[a:b])
        return Z
    return load


# ---------------------------------------------------------------------------
# LD pruning
# ---------------------------------------------------------------------------

@njit(cache=True)
def _prune_block(r2, in_window, row0, col0, keep, maf, threshold):
    """Greedy step-1 sliding-window pruning over one block (in position order).

    For each still-kept marker i, every later kept marker j in its window with
    r2 > threshold triggers removal of the lower-MAF member of the pair (ties
    remove the later marker). If i is removed, its scan stops.
    """
    n_rows, n_cols = r2.shape
    for ii in range(n_rows):
        i = row0 + ii
        if not keep[i]:
            continue
        for jj in range(n_cols):
            if not in_window[ii, jj]:
                continue
            j = col0 + jj
            if not keep[j]:
                continue
            if r2[ii, jj] > threshold:
                if maf[i] < maf[j]:
                    keep[i] = False
                    break
                keep[j] = False


def ld_prune(
    genotype: GenotypeLike,
    chromosomes: Sequence,
    positions: Sequence,
    *,
    r2_threshold: float = 0.2,
    window_kb: float = 500.0,
    max_window_snps: int = 500,
    min_maf: float = 0.01,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """Select an approximately LD-independent marker subset using in-sample r^2.

    Within each chromosome markers are visited in position order; for every
    pair closer than ``window_kb`` (and at most ``max_window_snps`` markers
    apart) with ``r^2 > r2_threshold`` the lower-MAF marker is dropped. This is
    the step-1 limit of PLINK ``--indep-pairwise``.

    Returns:
        Sorted array of retained global marker indices.
    """
    if not 0.0 < r2_threshold < 1.0:
        raise ValueError("r2_threshold must be in (0, 1)")
    if window_kb <= 0 or max_window_snps < 1:
        raise ValueError("window_kb and max_window_snps must be positive")
    pos_all = np.asarray(positions, dtype=np.float64)
    n_ind = _n_individuals(genotype)
    kept_global = []
    for _, gidx in chromosome_order(chromosomes, pos_all).items():
        if gidx.size == 0:
            continue
        # MAF filter first: only common, polymorphic markers enter pruning.
        maf = np.empty(gidx.size, dtype=np.float64)
        for s in range(0, gidx.size, 20000):
            _, maf[s:s + 20000] = standardized_columns(genotype, gidx[s:s + 20000])
        passing = maf >= max(min_maf, 1e-12)
        gidx = gidx[passing]
        maf = maf[passing]
        if gidx.size == 0:
            continue
        keep = np.ones(gidx.size, dtype=np.bool_)
        pos = pos_all[gidx]
        for blk in _iter_forward_blocks(
            _make_loader(genotype, gidx), pos, n_ind,
            window_kb * 1000.0, max_window_snps, chunk,
        ):
            r2 = (blk.r * blk.r).astype(np.float64)
            _prune_block(r2, blk.in_window, blk.rows[0], blk.cols[0], keep, maf, r2_threshold)
        kept_global.append(gidx[keep])
    if not kept_global:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate(kept_global)).astype(np.int64)


def pairwise_r2(genotype: GenotypeLike, index_marker: int, other_markers: np.ndarray) -> np.ndarray:
    """In-sample r^2 between one marker and a set of other markers."""
    idx = np.concatenate([[int(index_marker)], np.asarray(other_markers, dtype=np.int64)])
    Z, _ = standardized_columns(genotype, idx)
    r = (Z[:, 1:].T @ Z[:, 0]) / np.float32(Z.shape[0])
    return (r.astype(np.float64)) ** 2


# ---------------------------------------------------------------------------
# LD decay
# ---------------------------------------------------------------------------

@dataclass
class LDDecay:
    """Distribution of in-sample r^2 between marker pairs, by physical distance.

    Attributes:
        bin_start_kb, bin_end_kb: distance bins (log-spaced)
        mean_r2: mean r^2 of pairs in each bin (NaN when empty)
        upper_r2: ``quantile`` of r^2 in each bin (NaN when empty)
        n_pairs: number of pairs in each bin
        quantile: the quantile reported in ``upper_r2``
    """
    bin_start_kb: np.ndarray
    bin_end_kb: np.ndarray
    mean_r2: np.ndarray
    upper_r2: np.ndarray
    n_pairs: np.ndarray
    quantile: float

    def decay_distance_kb(self, r2_threshold: float) -> float:
        """Distance beyond which the upper r^2 quantile stays below ``r2_threshold``.

        Returns the start of the first bin from which every later non-empty bin
        is below the threshold, or the last bin end if LD never decays that far.
        """
        filled = np.flatnonzero(self.n_pairs > 0)
        if filled.size == 0:
            return float("nan")
        above = filled[self.upper_r2[filled] >= r2_threshold]
        if above.size == 0:
            return float(self.bin_start_kb[filled[0]])
        last = above.max()
        return float(self.bin_end_kb[last])

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({
            "Bin_Start_kb": self.bin_start_kb,
            "Bin_End_kb": self.bin_end_kb,
            "Mean_r2": self.mean_r2,
            f"Q{int(round(self.quantile * 100))}_r2": self.upper_r2,
            "N_Pairs": self.n_pairs,
        })


def ld_decay(
    genotype: GenotypeLike,
    chromosomes: Sequence,
    positions: Sequence,
    *,
    max_kb: float = 10_000.0,
    min_kb: float = 1.0,
    n_bins: int = 40,
    quantile: float = 0.9,
    max_window_snps: int = 2000,
    max_markers_per_chrom: int = 4000,
    min_maf: float = 0.05,
    seed: int = 0,
    chunk: int = DEFAULT_CHUNK,
) -> LDDecay:
    """Estimate LD decay with distance from the analysed genotypes.

    Pairs within ``max_kb`` (and at most ``max_window_snps`` markers apart) are
    binned on a log distance scale. Up to ``max_markers_per_chrom`` markers
    with MAF >= ``min_maf`` are sampled per chromosome, which bounds the cost
    for dense marker sets. r^2 quantiles are taken from a 1000-bin histogram
    per distance bin.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    pos_all = np.asarray(positions, dtype=np.float64)
    n_ind = _n_individuals(genotype)
    edges_kb = np.concatenate([[0.0], np.geomspace(min_kb, max_kb, n_bins)])
    edges_bp = edges_kb * 1000.0
    n_dist = edges_kb.size - 1
    hist = np.zeros((n_dist, 1000), dtype=np.int64)
    sums = np.zeros(n_dist)
    rng = np.random.default_rng(seed)

    for _, gidx in chromosome_order(chromosomes, pos_all).items():
        if gidx.size < 2:
            continue
        maf = np.empty(gidx.size, dtype=np.float64)
        for s in range(0, gidx.size, 20000):
            _, maf[s:s + 20000] = standardized_columns(genotype, gidx[s:s + 20000])
        gidx = gidx[maf >= min_maf]
        if gidx.size > max_markers_per_chrom:
            keep = np.sort(rng.choice(gidx.size, size=max_markers_per_chrom, replace=False))
            gidx = gidx[keep]
        if gidx.size < 2:
            continue
        pos = pos_all[gidx]
        for blk in _iter_forward_blocks(
            _make_loader(genotype, gidx), pos, n_ind, max_kb * 1000.0, max_window_snps, chunk,
        ):
            dist = (pos[blk.cols][np.newaxis, :] - pos[blk.rows][:, np.newaxis])[blk.in_window]
            r2 = np.clip((blk.r.astype(np.float64) ** 2)[blk.in_window], 0.0, 1.0)
            if dist.size == 0:
                continue
            dbin = np.clip(np.searchsorted(edges_bp, dist, side="right") - 1, 0, n_dist - 1)
            rbin = np.minimum((r2 * 1000).astype(np.int64), 999)
            np.add.at(hist, (dbin, rbin), 1)
            np.add.at(sums, dbin, r2)

    n_pairs = hist.sum(axis=1)
    mean_r2 = np.where(n_pairs > 0, sums / np.maximum(n_pairs, 1), np.nan)
    cum = np.cumsum(hist, axis=1)
    upper = np.full(n_dist, np.nan)
    for b in np.flatnonzero(n_pairs > 0):
        k = int(np.searchsorted(cum[b], quantile * n_pairs[b], side="left"))
        upper[b] = (k + 1) / 1000.0
    return LDDecay(edges_kb[:-1], edges_kb[1:], mean_r2, upper, n_pairs, float(quantile))
