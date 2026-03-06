"""
LOCO (Leave-One-Chromosome-Out) kinship matrix utilities.

This module is intentionally standalone so it can be removed cleanly if LOCO
is not adopted.
"""

from typing import Dict, List, Optional, Union, Tuple
import os
import numpy as np
import warnings
import pandas as pd

from ..utils.data_types import (
    CHROM_COLUMN,
    GenotypeMatrix,
    GenotypeMap,
    KinshipMatrix,
    attach_genotype_map_metadata,
    ensure_eager_genotype,
    group_marker_indices_by_labels,
)

# Check for joblib availability
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def _extract_chromosomes(map_data: Union[GenotypeMap, pd.DataFrame, np.ndarray, List],
                         n_markers: int) -> np.ndarray:
    """Extract chromosome labels aligned to genotype markers."""
    if isinstance(map_data, GenotypeMap):
        chroms = map_data.chromosomes
    elif isinstance(map_data, pd.DataFrame):
        if "CHROM" not in map_data.columns:
            raise ValueError("map_data is missing required column 'CHROM'")
        chroms = map_data["CHROM"]
    elif hasattr(map_data, "to_dataframe"):
        map_df = map_data.to_dataframe()
        if "CHROM" not in map_df.columns:
            raise ValueError("map_data is missing required column 'CHROM'")
        chroms = map_df["CHROM"]
    else:
        chroms = np.asarray(map_data)

    chroms = np.asarray(chroms).astype(str, copy=False)
    if chroms.ndim != 1 or len(chroms) != n_markers:
        raise ValueError("Chromosome labels must be a 1D array aligned to genotype markers")
    return chroms


def _group_markers_by_chrom(chrom_values: np.ndarray) -> Dict[str, np.ndarray]:
    """Return ordered marker indices grouped by chromosome.

    Uses vectorized numpy operations for speed instead of Python loops.
    """
    return group_marker_indices_by_labels(chrom_values)


def _resolve_chromosome_groups(
    map_data: Union[GenotypeMap, pd.DataFrame, np.ndarray, List],
    n_markers: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """Resolve chromosome labels and cached marker-group indices."""
    chrom_values = _extract_chromosomes(map_data, n_markers)

    if isinstance(map_data, GenotypeMap):
        chrom_groups = map_data.get_chromosome_groups()
        chrom_order = map_data.get_chromosome_order()
    elif isinstance(map_data, pd.DataFrame):
        if CHROM_COLUMN not in map_data.columns:
            raise ValueError("map_data is missing required column 'CHROM'")
        attach_genotype_map_metadata(map_data)
        chrom_groups = map_data.attrs.get("chromosome_groups") or _group_markers_by_chrom(chrom_values)
        chrom_order = list(map_data.attrs.get("chromosome_order") or chrom_groups.keys())
    elif hasattr(map_data, "to_dataframe"):
        map_df = map_data.to_dataframe()
        if CHROM_COLUMN not in map_df.columns:
            raise ValueError("map_data is missing required column 'CHROM'")
        attach_genotype_map_metadata(map_df)
        chrom_groups = map_df.attrs.get("chromosome_groups") or _group_markers_by_chrom(chrom_values)
        chrom_order = list(map_df.attrs.get("chromosome_order") or chrom_groups.keys())
    else:
        chrom_groups = _group_markers_by_chrom(chrom_values)
        chrom_order = list(chrom_groups.keys())

    normalized_groups: Dict[str, np.ndarray] = {}
    total_markers = 0
    for chrom in chrom_order:
        indices = np.asarray(chrom_groups[str(chrom)], dtype=np.int64)
        normalized_groups[str(chrom)] = indices
        total_markers += int(indices.size)

    if total_markers != n_markers:
        raise ValueError("Chromosome groups must cover all genotype markers exactly once")

    return chrom_values, normalized_groups, [str(chrom) for chrom in chrom_order]


def _get_genotype_columns(
    genotype: Union[GenotypeMatrix, np.ndarray],
    indices: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Fetch marker columns in individual-major order without mutating source storage."""
    if isinstance(genotype, GenotypeMatrix):
        return np.array(genotype.get_columns(indices), dtype=dtype, copy=True)
    return np.array(genotype[:, indices], dtype=dtype, copy=True)


class LocoKinship:
    """Container for LOCO kinship computations and cached eigendecompositions."""

    def __init__(self,
                 total_raw: np.ndarray,
                 total_diag: np.ndarray,
                 chrom_raw: Dict[str, np.ndarray],
                 chrom_diag: Dict[str, np.ndarray],
                 chrom_order: List[str]):
        self._total_raw = total_raw
        self._total_diag = total_diag
        self._chrom_raw = chrom_raw
        self._chrom_diag = chrom_diag
        self._chrom_order = list(chrom_order)

        self._loco_cache: Dict[str, KinshipMatrix] = {}
        self._eigen_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._full_cache: Optional[KinshipMatrix] = None

    @property
    def chromosomes(self) -> List[str]:
        """Chromosome labels in the order they appeared."""
        return list(self._chrom_order)

    def _normalize(self, raw: np.ndarray, diag: np.ndarray, label: str) -> KinshipMatrix:
        """Symmetrize and normalize a raw kinship matrix."""
        kin = (raw + raw.T) / 2.0
        mean_diag = float(np.mean(diag))
        if mean_diag > 0:
            kin = kin / mean_diag
        else:
            warnings.warn(f"Mean diagonal for {label} is non-positive; skipping normalization")
        return KinshipMatrix(kin)

    def get_full(self) -> KinshipMatrix:
        """Return the full (non-LOCO) kinship matrix."""
        if self._full_cache is None:
            self._full_cache = self._normalize(self._total_raw, self._total_diag, "full")
        return self._full_cache

    def get_loco(self, chrom: Union[str, int]) -> KinshipMatrix:
        """Return the LOCO kinship matrix for a chromosome."""
        chrom_key = str(chrom)
        if chrom_key in self._loco_cache:
            return self._loco_cache[chrom_key]
        if chrom_key not in self._chrom_raw:
            raise KeyError(f"Chromosome {chrom_key} not found in LOCO kinship")

        raw_loco = self._total_raw - self._chrom_raw[chrom_key]
        diag_loco = self._total_diag - self._chrom_diag[chrom_key]
        kin = self._normalize(raw_loco, diag_loco, f"loco:{chrom_key}")
        self._loco_cache[chrom_key] = kin
        return kin

    def get_eigen(self, chrom: Union[str, int]) -> Dict[str, np.ndarray]:
        """Return cached eigendecomposition for a LOCO kinship matrix.

        Eigendecomposition is performed in float64 for numerical stability,
        but eigenvectors are stored as float32 C-contiguous for faster downstream matmul.
        """
        chrom_key = str(chrom)
        if chrom_key in self._eigen_cache:
            return self._eigen_cache[chrom_key]

        # Eigendecomposition in float64 for numerical stability
        kinship = self.get_loco(chrom_key).to_numpy().astype(np.float64)
        eigenvals, eigenvecs = np.linalg.eigh(kinship)
        sort_indices = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sort_indices]
        eigenvecs = eigenvecs[:, sort_indices]

        # Store eigenvectors as float32 C-contiguous for faster MLM crossproducts
        # np.ascontiguousarray ensures C-order which is optimal for eigenvecs.T @ G_batch
        eigen = {"eigenvals": eigenvals, "eigenvecs": np.ascontiguousarray(eigenvecs.astype(np.float32))}
        self._eigen_cache[chrom_key] = eigen
        return eigen


def _compute_chrom_kinship(chrom: str,
                           indices: np.ndarray,
                           genotype: Union[GenotypeMatrix, np.ndarray],
                           n_individuals: int,
                           is_imputed: bool) -> Tuple[str, np.ndarray, np.ndarray]:
    """Compute kinship contribution for a single chromosome.

    This function is designed to be called in parallel.
    Uses float32 for faster matmul operations.

    Returns:
        Tuple of (chrom, raw_kinship, diag) as float32 arrays
    """
    # Get genotype data for this chromosome (float32 for faster matmul)
    Z = _get_genotype_columns(genotype, indices, dtype=np.float32)

    if is_imputed:
        means = np.mean(Z, axis=0)
    else:
        # Convert -9 to NaN so nanmean excludes them from mean calculation.
        missing_mask = (Z == -9) | np.isnan(Z)
        if missing_mask.any():
            Z[missing_mask] = np.nan
        means = np.nanmean(Z, axis=0)
        means[np.isnan(means)] = 0.0
    Z -= means[np.newaxis, :]

    # Replace missing values with 0 (centered mean) for kinship calculation.
    if not is_imputed and not np.all(np.isfinite(Z)):
        Z[~np.isfinite(Z)] = 0.0

    # Compute raw kinship contribution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = Z @ Z.T
    diag = np.sum(Z * Z, axis=1)

    return chrom, raw, diag


def PANICLE_K_VanRaden_LOCO(M: Union[GenotypeMatrix, np.ndarray],
                        map_data: Union[GenotypeMap, pd.DataFrame, np.ndarray, List],
                        maxLine: int = 5000,
                        cpu: int = 1,
                        verbose: bool = True) -> LocoKinship:
    """Compute LOCO kinship using VanRaden-style raw cross-products.

    Args:
        M: Genotype matrix (n_individuals × n_markers)
        map_data: Genetic map with chromosome information
        maxLine: Batch size for processing (used in sequential mode)
        cpu: Number of CPU cores for parallel chromosome processing
        verbose: Print progress information

    Returns:
        LocoKinship object with total and per-chromosome kinship data
    """
    M = ensure_eager_genotype(M)

    if isinstance(M, GenotypeMatrix):
        genotype_data = M
        n_individuals = M.n_individuals
        n_markers = M.n_markers
        is_imputed = M.is_imputed
    elif isinstance(M, np.ndarray):
        genotype_data = M
        n_individuals, n_markers = M.shape
        is_imputed = False  # Raw numpy arrays need -9 checks
    else:
        raise ValueError("M must be GenotypeMatrix or numpy array")

    _, chrom_groups, chrom_order = _resolve_chromosome_groups(map_data, n_markers)
    n_chroms = len(chrom_order)

    if verbose:
        print(f"Calculating LOCO kinship for {n_individuals} individuals, {n_markers} markers")
        print(f"Chromosomes: {n_chroms}")

    # Handle cpu=0 to mean use all available cores
    if cpu == 0:
        import multiprocessing
        cpu = multiprocessing.cpu_count()

    # Determine if we should use parallel processing.
    # Empirical runs on real sorghum datasets (170k to 4.2M markers) showed
    # threading is typically slower for LOCO kinship due memory-bandwidth limits.
    # Keep a high default threshold and expose an explicit override.
    n_workers = min(cpu, n_chroms)
    force_parallel = os.getenv("PANICLE_FORCE_LOCO_KINSHIP_PARALLEL", "").lower() in {
        "1", "true", "yes", "on",
    }
    parallel_worthwhile = n_workers > 1 and n_markers >= 8_000_000 and n_chroms >= 3
    if force_parallel:
        parallel_worthwhile = n_workers > 1 and n_chroms >= 2
    use_parallel = HAS_JOBLIB and parallel_worthwhile

    if use_parallel:
        if verbose:
            suffix = " (forced by PANICLE_FORCE_LOCO_KINSHIP_PARALLEL)" if force_parallel else ""
            print(f"Using parallel processing with {n_workers} workers{suffix}")

        # Use threading to avoid expensive serialization of genotype arrays.
        results = Parallel(
            n_jobs=n_workers,
            backend='threading',
            pre_dispatch=n_workers,
            batch_size=1,
        )(
            delayed(_compute_chrom_kinship)(chrom, indices, genotype_data, n_individuals, is_imputed)
            for chrom, indices in chrom_groups.items()
        )

        # Aggregate results (float32 for consistency)
        raw_by_chrom = {}
        diag_by_chrom = {}
        raw_total = np.zeros((n_individuals, n_individuals), dtype=np.float32)
        diag_total = np.zeros(n_individuals, dtype=np.float32)

        for chrom, raw, diag in results:
            raw_by_chrom[chrom] = (raw + raw.T) / 2.0  # Symmetrize
            diag_by_chrom[chrom] = diag
            raw_total += raw
            diag_total += diag

        raw_total = (raw_total + raw_total.T) / 2.0

    else:
        # Sequential processing - optimized to process by chromosome
        # This avoids redundant total matmul and per-batch chromosome splitting
        if verbose and not HAS_JOBLIB and cpu > 1:
            print("Note: joblib not available, using sequential processing")
        elif verbose and cpu > 1 and not parallel_worthwhile:
            print("Parallel overhead likely exceeds benefit for this workload; using sequential processing")

        raw_by_chrom = {}
        diag_by_chrom = {}

        # Process each chromosome separately
        for chrom_idx, chrom in enumerate(chrom_order):
            indices = chrom_groups[chrom]
            n_chrom_markers = len(indices)

            if verbose:
                print(f"Processing chromosome {chrom} ({n_chrom_markers} markers)")

            # Initialize accumulator for this chromosome (float32 for faster matmul)
            raw_chrom = np.zeros((n_individuals, n_individuals), dtype=np.float32)
            diag_chrom = np.zeros(n_individuals, dtype=np.float32)

            # Process chromosome markers in batches
            n_chrom_batches = (n_chrom_markers + maxLine - 1) // maxLine
            for batch_idx in range(n_chrom_batches):
                start_idx = batch_idx * maxLine
                end_idx = min(start_idx + maxLine, n_chrom_markers)
                batch_indices = indices[start_idx:end_idx]

                # Get genotype data for this batch (float32 for faster matmul)
                Z_batch = _get_genotype_columns(genotype_data, batch_indices, dtype=np.float32)

                # Handle missing values only if data is not pre-imputed
                if is_imputed:
                    # Data is pre-imputed, just use regular mean
                    means_batch = np.mean(Z_batch, axis=0)
                else:
                    # Handle missing values: -9 sentinel and NaN
                    # Convert -9 to NaN so nanmean excludes them from mean calculation
                    missing_mask = (Z_batch == -9) | np.isnan(Z_batch)
                    if missing_mask.any():
                        Z_batch[missing_mask] = np.nan

                    # Center by column means (nanmean excludes NaN/missing)
                    means_batch = np.nanmean(Z_batch, axis=0)
                    means_batch[np.isnan(means_batch)] = 0.0

                Z_batch -= means_batch[np.newaxis, :]

                # Replace any remaining non-finite values with 0
                if not is_imputed and not np.all(np.isfinite(Z_batch)):
                    Z_batch[~np.isfinite(Z_batch)] = 0.0

                # Accumulate kinship contribution (guard against spurious BLAS FPE flags)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        raw_chrom += Z_batch @ Z_batch.T
                diag_chrom += np.sum(Z_batch * Z_batch, axis=1)

            # Symmetrize and store (keep as float32, will convert for eigendecomp)
            raw_by_chrom[chrom] = (raw_chrom + raw_chrom.T) / 2.0
            diag_by_chrom[chrom] = diag_chrom

        # Compute total from per-chromosome sums (avoids redundant computation)
        # Keep as float32 for consistency; eigendecomp will convert to float64
        raw_total = np.zeros((n_individuals, n_individuals), dtype=np.float32)
        diag_total = np.zeros(n_individuals, dtype=np.float32)
        for chrom in chrom_order:
            raw_total += raw_by_chrom[chrom]
            diag_total += diag_by_chrom[chrom]

    return LocoKinship(
        total_raw=raw_total,
        total_diag=diag_total,
        chrom_raw=raw_by_chrom,
        chrom_diag=diag_by_chrom,
        chrom_order=chrom_order,
    )
