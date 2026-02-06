"""
FarmCPU (Fixed and random model Circulating Probability Unification) for GWAS.

This module provides the PANICLE_FarmCPU function which implements an iterative
multi-locus GWAS method that combines GLM and MLM approaches.

Based on the rMVP R implementation by Xiaolei Liu and Zhiwu Zhang.

Memory Efficiency Notes:
------------------------
This implementation is optimized for large datasets (5M+ markers):

1. Streaming genotype processing: Genotype data is processed in batches via GLM,
   avoiding the need to load the entire matrix into memory. Memory-mapped
   GenotypeMatrix objects are preserved without materialization.

2. Efficient binning: The _farmcpu_specify function uses in-place operations
   to minimize temporary array allocations.

3. Memory requirements scale with:
   - P-values array: O(n_markers) float64 (~40MB per 5M markers)
   - Effects/SE arrays: O(n_markers) float64 each
   - Temporary sorting arrays: O(n_markers) during binning
   - Pseudo-QTN genotypes: O(n_individuals × n_qtns) - typically small

For 5M markers with 10k individuals, expect ~200-400MB working memory
(vs 200GB+ in the unoptimized version that pre-loaded all genotypes).
"""

from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
from scipy import linalg, special, stats

from ..utils.data_types import GenotypeMatrix, GenotypeMap, AssociationResults
from .glm import PANICLE_GLM


def _pvalue_to_t_critical(p_threshold: float) -> float:
    """Convert a p-value threshold to a critical |t| value.

    Uses the inverse of the normal approximation: t = sqrt(2) * erfcinv(p)
    This matches the forward computation in GLM: p = erfc(|t| / sqrt(2))

    Args:
        p_threshold: Two-tailed p-value threshold (e.g., 0.01)

    Returns:
        Critical |t| value such that |t| > t_critical implies p < p_threshold
    """
    # Clamp to avoid numerical issues
    p_threshold = max(p_threshold, 1e-300)
    p_threshold = min(p_threshold, 1.0)
    return np.sqrt(2.0) * special.erfcinv(p_threshold)


def _t_stat_to_pvalue(t_stats: np.ndarray) -> np.ndarray:
    """Convert |t|-statistics to two-tailed p-values.

    Uses the normal approximation: p = erfc(|t| / sqrt(2))

    Args:
        t_stats: Array of absolute t-statistics

    Returns:
        Two-tailed p-values
    """
    p = special.erfc(t_stats / np.sqrt(2.0))
    return np.clip(p, 0.0, 1.0)

# rMVP preprocessing replaces missing genotypes (-9/NA) with the heterozygote dosage (1).
# Use the same fill value so that GLM/FarmCPU statistics remain comparable.
MISSING_FILL_VALUE = 1.0


def _numeric_chromosomes(chrom_series: "pd.Series") -> np.ndarray:
    """Convert chromosome labels to numeric values (preserve relative ordering)."""
    import pandas as pd  # local import to avoid hard dependency at import time

    # Check if dtype is numeric, handling pandas extension dtypes (e.g., StringDtype)
    # that numpy doesn't understand
    try:
       is_numeric = np.issubdtype(chrom_series.dtype, np.number)
    except TypeError:
       # pandas StringDtype or other non-numpy dtype
       is_numeric = False
       
    if is_numeric:
       return chrom_series.to_numpy(dtype=float, copy=False)

    chrom_as_str = chrom_series.astype(str)
    chrom_order = sorted(chrom_as_str.unique(), key=lambda val: (len(val), val))
    chrom_lookup = {val: float(idx) for idx, val in enumerate(chrom_order)}
    return chrom_as_str.map(chrom_lookup).to_numpy(dtype=float)


# =============================================================================
# Data Structure Documentation
# =============================================================================
#
# GenotypeMatrix:
#     Memory-efficient genotype matrix with lazy loading support.
#     - Shape: (n_individuals, n_markers)
#     - Dtype: typically int8 for storage, converts to float for computation
#     - Missing values: encoded as -9, imputed via major allele by default
#     - Key properties:
#         .shape -> (n_individuals, n_markers)
#         .n_individuals -> int
#         .n_markers -> int
#         .is_imputed -> bool (whether -9 values have been pre-imputed)
#     - Key methods:
#         .get_marker(idx) -> np.ndarray of shape (n_individuals,)
#         .get_batch(start, end) -> np.ndarray of shape (n_individuals, end-start)
#         .get_marker_imputed(idx, dtype=np.float64) -> imputed marker as float
#         .get_batch_imputed(start, end, dtype=np.float64) -> imputed batch
#         .get_columns_imputed(indices) -> arbitrary columns with imputation
#         .subset_individuals(indices) -> new GenotypeMatrix for subset
#
# GenotypeMap:
#     SNP map information with chromosome and position data.
#     - Required columns in underlying DataFrame: ['SNP', 'CHROM', 'POS']
#     - Optional columns: 'REF', 'ALT'
#     - Key properties:
#         .snp_ids -> pd.Series of SNP identifiers
#         .chromosomes -> pd.Series of chromosome values
#         .positions -> pd.Series of physical positions (bp)
#         .n_markers -> int
#     - Key methods:
#         .to_dataframe() -> pd.DataFrame copy
#         .with_metadata(**kwargs) -> new GenotypeMap with additional metadata
#     - Metadata dictionary (.metadata) can store arbitrary key-value pairs
#
# AssociationResults:
#     GWAS association results container.
#     - Attributes:
#         .effects -> np.ndarray of effect sizes (beta coefficients)
#         .se -> np.ndarray of standard errors
#         .pvalues -> np.ndarray of p-values
#         .snp_map -> Optional[GenotypeMap] for SNP annotations
#         .n_markers -> int
#     - Key methods:
#         .to_dataframe() -> pd.DataFrame with columns:
#             If snp_map provided: ['SNP', 'Chr', 'Pos', 'Effect', 'SE', 'P-value']
#             Otherwise: ['Effect', 'SE', 'P-value']
#         .to_numpy() -> np.ndarray of shape (n_markers, 3) [Effect, SE, P-value]
#
# =============================================================================
# PANICLE_GLM Documentation
# =============================================================================
#
# PANICLE_GLM(phe, geno, CV=None, maxLine=5000, cpu=1, verbose=True,
#             impute_missing=True, major_alleles=None, missing_fill_value=1.0)
#
# General Linear Model for GWAS using optimized FWL+QR algorithm.
#
# Inputs:
#     phe: np.ndarray
#         Phenotype array of shape (n_individuals, 2)
#         Column 0: Individual IDs (can be string or numeric)
#         Column 1: Trait values (numeric)
#
#     geno: Union[GenotypeMatrix, np.ndarray]
#         Genotype matrix of shape (n_individuals, n_markers)
#         Values typically 0, 1, 2 representing allele dosage
#         Missing values encoded as -9
#
#     CV: Optional[np.ndarray]
#         Covariate matrix of shape (n_individuals, n_covariates)
#         Does NOT include intercept (added internally)
#         Can include PCs, population structure, experimental covariates
#         If None, only intercept is used
#
#     maxLine: int
#         Batch size for processing markers (default 5000)
#         Larger values use more memory but may be faster
#
#     cpu: int
#         Unused, kept for API compatibility
#
#     verbose: bool
#         Print progress information
#
#     missing_fill_value: float
#         Value to impute for missing genotypes (default 1.0 = heterozygote)
#
# Output:
#     AssociationResults object containing:
#         .effects: Effect sizes (beta) for each marker
#         .se: Standard errors for each effect
#         .pvalues: P-values from t-test for each marker
#         .snp_map: None (not set by GLM, must be added separately)
#
# =============================================================================


def _validate_genotype(
    geno: Union[GenotypeMatrix, np.ndarray],
    verbose: bool = True
) -> Union[GenotypeMatrix, np.ndarray]:
    """Validate genotype input without loading entire matrix into memory.

    For large datasets (5M+ markers), we avoid pre-loading the entire matrix.
    The GLM already handles batched imputation efficiently via get_batch_imputed().

    Args:
        geno: Input genotype matrix (GenotypeMatrix or numpy array)
        verbose: Print progress information

    Returns:
        The validated genotype matrix (unchanged for GenotypeMatrix,
        wrapped for numpy arrays if needed)
    """
    if isinstance(geno, GenotypeMatrix):
        # Keep memory-mapped/lazy-loaded data as-is
        # GLM will handle batched imputation efficiently
        if verbose and not geno.is_imputed:
            print("FarmCPU: Using streaming mode for genotype processing (memory-efficient)")
        return geno

    elif isinstance(geno, np.ndarray):
        # For numpy arrays, wrap in GenotypeMatrix for consistent interface
        # but don't copy or transform - let GLM handle imputation in batches
        if verbose:
            print("FarmCPU: Wrapping numpy array in GenotypeMatrix interface")
        # precompute_alleles=False avoids scanning entire matrix
        return GenotypeMatrix(geno, is_imputed=False, precompute_alleles=False)

    raise ValueError("Unsupported genotype type")

def _farmcpu_specify(
    map_data: GenotypeMap,
    P: np.ndarray,
    bin_size: int,
    inclosure_size: int,
    max_bp: float = 1e10,
    use_t_stats: bool = False,
    _cached_bin_ids: np.ndarray = None
) -> np.ndarray:
    """Identify representative SNPs from genomic bins.

    Optimized strategy using argsort + segment-wise reduction (~17x faster than lexsort):
    1. Sort markers by bin_id only (not lexsort on two keys)
    2. Use reduceat to find best stat per bin in O(n)
    3. Find marker indices for best stats using vectorized operations

    Args:
        map_data: SNP map with chromosome and position info
        P: P-values (or |t|-statistics if use_t_stats=True) for all markers
        bin_size: Size of genomic bins in base pairs
        inclosure_size: Maximum number of bins to select
        max_bp: Maximum base pairs for creating unique IDs
        use_t_stats: If True, P contains |t|-statistics (larger = more significant)
                     instead of p-values (smaller = more significant)
        _cached_bin_ids: Pre-computed bin IDs (for repeated calls with same bin_size)

    Returns:
        Indices of selected pseudo-QTNs
    """
    n_markers = len(P)

    # Compute bin IDs (or use cached)
    if _cached_bin_ids is not None:
        bin_ids = _cached_bin_ids
    else:
        chromosomes = _numeric_chromosomes(map_data.chromosomes)
        positions = map_data.positions.values.astype(np.float64)
        bin_ids = np.floor((positions + chromosomes * max_bp) / bin_size).astype(np.int64)

    # Sort markers by bin_id only (much faster than lexsort on two keys)
    bin_order = np.argsort(bin_ids, kind='mergesort')
    sorted_bins = bin_ids[bin_order]
    sorted_stats = P[bin_order]

    # Find unique bins and their boundaries
    unique_bins, bin_starts = np.unique(sorted_bins, return_index=True)
    n_bins = len(unique_bins)

    if n_bins == 0:
        return np.array([], dtype=np.int64)

    # Get best stat per bin using reduceat (O(n) operation)
    if use_t_stats:
        # For t-stats: larger = more significant, use maximum
        best_stat_per_bin = np.maximum.reduceat(sorted_stats, bin_starts)
    else:
        # For p-values: smaller = more significant, use minimum
        best_stat_per_bin = np.minimum.reduceat(sorted_stats, bin_starts)

    # Find which marker has the best stat in each bin (vectorized)
    bin_ends = np.concatenate([bin_starts[1:], [n_markers]])
    bin_lengths = bin_ends - bin_starts
    bin_membership = np.repeat(np.arange(n_bins), bin_lengths)

    # Mask: is this marker's stat equal to the best for its bin?
    is_best = sorted_stats == best_stat_per_bin[bin_membership]

    # Get first occurrence of best in each bin using cumsum with resets
    cumsum = np.cumsum(is_best.astype(np.int32))
    cumsum_before = np.zeros(n_markers, dtype=np.int32)
    cumsum_before[bin_starts] = cumsum[bin_starts] - is_best[bin_starts].astype(np.int32)
    cumsum_before = np.maximum.accumulate(cumsum_before)
    rank_in_bin = cumsum - cumsum_before

    # First best in each bin has rank == 1
    first_best = is_best & (rank_in_bin == 1)
    rep_sorted_idx = np.where(first_best)[0]
    representative_indices = bin_order[rep_sorted_idx]
    representative_stats = P[representative_indices]

    # Select top inclosure_size bins by significance
    n_select = min(inclosure_size, len(representative_indices))

    if n_select == 0:
        return np.array([], dtype=np.int64)

    if n_select >= len(representative_indices):
        return representative_indices.copy()

    # Use argpartition for O(n) top-k selection
    if use_t_stats:
        top_k_idx = np.argpartition(-representative_stats, n_select)[:n_select]
    else:
        top_k_idx = np.argpartition(representative_stats, n_select)[:n_select]

    return representative_indices[top_k_idx].copy()


def _farmcpu_remove(
    geno: Union[GenotypeMatrix, np.ndarray],
    map_data: GenotypeMap,
    seq_qtn: np.ndarray,
    seq_qtn_p: np.ndarray,
    threshold: float = 0.7,
    max_samples: int = 100000,
    use_t_stats: bool = False
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Remove pseudo-QTNs that are highly correlated (LD pruning).

    Args:
        geno: Genotype matrix (n_individuals × n_markers)
        map_data: SNP map information
        seq_qtn: Indices of candidate pseudo-QTNs
        seq_qtn_p: P-values (or |t|-statistics if use_t_stats=True) for candidate pseudo-QTNs
        threshold: Correlation threshold for removing markers
        max_samples: Maximum number of individuals to use for correlation
        use_t_stats: If True, seq_qtn_p contains |t|-statistics (larger = more significant)

    Returns:
        Tuple of (bin genotypes for selected QTNs, updated QTN indices)
    """
    if seq_qtn is None or len(seq_qtn) == 0:
        return None, np.array([], dtype=int)

    # Sort QTNs by significance (keep most significant)
    # For p-values: ascending (lower = more significant)
    # For t-stats: descending (higher = more significant)
    if use_t_stats:
        order = np.argsort(-seq_qtn_p)  # Descending for t-stats
    else:
        order = np.argsort(seq_qtn_p)   # Ascending for p-values
    seq_qtn = seq_qtn[order].copy()
    seq_qtn_p = seq_qtn_p[order].copy()

    # Filter by unique chromosome + position
    chromosomes = _numeric_chromosomes(map_data.chromosomes)
    positions = map_data.positions.values.astype(float)

    huge_num = 1e10
    cb = chromosomes[seq_qtn].astype(float) * huge_num + positions[seq_qtn].astype(float)
    _, unique_idx = np.unique(cb, return_index=True)
    unique_idx = np.sort(unique_idx)  # Preserve p-value order
    seq_qtn = seq_qtn[unique_idx]

    if len(seq_qtn) == 0:
        return None, np.array([], dtype=int)

    # Get genotypes for QTNs (subsample if needed)
    n_ind = geno.n_individuals if isinstance(geno, GenotypeMatrix) else geno.shape[0]
    n_samples = min(n_ind, max_samples)
    sample_idx = np.arange(n_samples)

    if isinstance(geno, GenotypeMatrix):
        x = geno.get_columns_imputed(seq_qtn, dtype=np.float32)[sample_idx, :]
    else:
        x = geno[sample_idx][:, seq_qtn].astype(np.float32)

    # Calculate correlation matrix and prune
    if x.shape[1] > 1:
        # Handle constant columns
        std = np.std(x, axis=0)
        constant_cols = std < 1e-10
        std[constant_cols] = 1  # Prevent division by zero

        x_centered = (x - np.mean(x, axis=0)) / std

        # Set constant columns to zero (no correlation)
        x_centered[:, constant_cols] = 0

        r = np.corrcoef(x_centered.T)
        if r.ndim == 0:
            r = np.array([[r]])

        # Replace NaN with 0 (can happen with constant columns)
        r = np.nan_to_num(r, nan=0.0)

        # Mark highly correlated pairs - keep the one with lower p-value
        keep = np.ones(len(seq_qtn), dtype=bool)
        for i in range(len(seq_qtn)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(seq_qtn)):
                if keep[j] and abs(r[i, j]) > threshold:
                    keep[j] = False

        seq_qtn = seq_qtn[keep]

    if len(seq_qtn) == 0:
        return None, np.array([], dtype=int)



    # if isinstance(genotype, GenotypeMatrix):
    #     precomputed_major_alleles = genotype.major_alleles
    # else:
       # ... legacy code ..._imputed(seq_qtn, dtype=np.float32)
    # Get final bin genotypes
    if isinstance(geno, GenotypeMatrix):
        bin_geno = geno.get_columns_imputed(seq_qtn, dtype=np.float32)
    else:
        bin_geno = geno[:, seq_qtn].astype(np.float32)

    return bin_geno, seq_qtn


def _farmcpu_bin(
    phe: np.ndarray,
    geno: Union[GenotypeMatrix, np.ndarray],
    map_data: GenotypeMap,
    CV: Optional[np.ndarray],
    P: np.ndarray,
    method: str,
    bin_sizes: List[int],
    bin_selections: List[int],
    the_loop: int,
    bound: Optional[int],
    verbose: bool = True,
    use_t_stats: bool = False
) -> np.ndarray:
    """Select pseudo-QTNs using binning strategy.

    Args:
        phe: Phenotype matrix
        geno: Genotype matrix
        map_data: SNP map information
        CV: Covariates
        P: Current p-values (or |t|-statistics if use_t_stats=True)
        method: Binning method ("static", "EMMA", "FaST-LMM")
        bin_sizes: List of bin sizes to try
        bin_selections: List of selection counts to try
        the_loop: Current iteration number (1-indexed)
        bound: Maximum number of pseudo-QTNs
        verbose: Print progress
        use_t_stats: If True, P contains |t|-statistics (larger = more significant)

    Returns:
        Indices of selected pseudo-QTNs
    """
    if P is None:
        return np.array([], dtype=int)

    # Handle 2D P-values (if return_cov_stats=True was used)
    # Binning should be based on the marker's own P-value (column 0)
    if P.ndim == 2:
        P = P[:, 0]

    n = phe.shape[0]

    # Set upper bound for bin selection (matches R: sqrt(n)/sqrt(log10(n)))
    if bound is None:
        bound = int(round(np.sqrt(n) / np.sqrt(np.log10(max(n, 10)))))

    # Filter selections to be within bound
    bin_selections = [min(s, bound) for s in bin_selections]
    bin_selections = sorted(set(s for s in bin_selections if s <= bound))
    if not bin_selections:
        bin_selections = [bound]

    optimizable = len(bin_sizes) * len(bin_selections) > 1

    if method == "static" or not optimizable:
        # Static method: use different bin sizes for different iterations
        # R: loop 2 uses b[3], loop 3 uses b[2], else uses b[1]
        if the_loop == 2:
            bin_size = bin_sizes[-1]  # Largest bin (5e7)
        elif the_loop == 3:
            bin_size = bin_sizes[1] if len(bin_sizes) > 1 else bin_sizes[0]
        else:
            bin_size = bin_sizes[0]  # Smallest bin (5e5)

        inc_size = bound

        if verbose:
            print(f"Optimizing Pseudo QTNs... (bin_size={bin_size}, max_qtns={inc_size})")

        seq_qtn = _farmcpu_specify(map_data, P, bin_size, inc_size, use_t_stats=use_t_stats)

    elif method == "EMMA" and optimizable:
        # EMMA method: optimize bin size and selection by REML
        if verbose:
            print("Optimizing Pseudo QTNs (EMMA)...")

        best_reml = np.inf
        best_seq_qtn = np.array([], dtype=int)

        for bin_size in bin_sizes:
            for inc_size in bin_selections:
                seq_qtn = _farmcpu_specify(map_data, P, bin_size, inc_size, use_t_stats=use_t_stats)

                if len(seq_qtn) == 0:
                    continue

                # Get QTN genotypes
                if isinstance(geno, GenotypeMatrix):
                    GK = geno.get_columns_imputed(seq_qtn, dtype=np.float32)
                else:
                    GK = geno[:, seq_qtn].astype(np.float32)

                # Calculate REML using EMMA
                reml = _farmcpu_burger_emma(phe, CV, GK)

                if verbose:
                    print(f"  bin={bin_size}, n={inc_size}, -2LL={reml:.2f}")

                if reml < best_reml:
                    best_reml = reml
                    best_seq_qtn = seq_qtn.copy()

        seq_qtn = best_seq_qtn

    elif method == "FaST-LMM" and optimizable:
        # FaST-LMM method
        if verbose:
            print("Optimizing Pseudo QTNs (FaST-LMM)...")

        best_reml = np.inf
        best_seq_qtn = np.array([], dtype=int)

        for bin_size in bin_sizes:
            for inc_size in bin_selections:
                seq_qtn = _farmcpu_specify(map_data, P, bin_size, inc_size, use_t_stats=use_t_stats)

                if len(seq_qtn) == 0:
                    continue

                if isinstance(geno, GenotypeMatrix):
                    GK = geno.get_columns_imputed(seq_qtn, dtype=np.float32)
                else:
                    GK = geno[:, seq_qtn].astype(np.float32)

                reml = _farmcpu_burger_fastlmm(phe, CV, GK)

                if verbose:
                    print(f"  bin={bin_size}, n={inc_size}, -2LL={reml:.2f}")

                if reml < best_reml:
                    best_reml = reml
                    best_seq_qtn = seq_qtn.copy()

        seq_qtn = best_seq_qtn
    else:
        seq_qtn = np.array([], dtype=int)

    return seq_qtn


def _farmcpu_burger_emma(
    phe: np.ndarray,
    CV: Optional[np.ndarray],
    GK: np.ndarray
) -> float:
    """Calculate -2 * log-likelihood using EMMA method.

    Args:
        phe: Phenotype matrix (n × 2)
        CV: Covariates (n × c) or None
        GK: Pseudo-QTN genotypes (n × s)

    Returns:
        -2 * REML log-likelihood
    """
    from ..matrix.kinship import PANICLE_K_VanRaden

    y = phe[:, 1].astype(np.float64)
    n = len(y)

    # Build covariate matrix
    if CV is not None:
        X = np.column_stack([np.ones(n), CV])
    else:
        X = np.ones((n, 1))

    # Compute kinship from pseudo-QTNs
    # PANICLE_K_VanRaden expects individuals × markers
    K_obj = PANICLE_K_VanRaden(GK, verbose=False)
    K = K_obj.to_numpy()

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

    # Grid search for delta
    best_ll = -np.inf
    for log_delta in np.arange(-5, 5.1, 0.1):
        delta = np.exp(log_delta)

        # Transform
        D_inv = 1.0 / (eigenvalues + delta)
        Ut_y = eigenvectors.T @ y
        Ut_X = eigenvectors.T @ X

        # Weighted least squares
        XtDX = Ut_X.T @ (D_inv[:, None] * Ut_X)
        XtDy = Ut_X.T @ (D_inv * Ut_y)

        try:
            beta = np.linalg.solve(XtDX, XtDy)
        except np.linalg.LinAlgError:
            continue

        residuals = Ut_y - Ut_X @ beta
        sigma2 = np.sum(D_inv * residuals**2) / n

        # Log-likelihood
        ll = -0.5 * (n * np.log(2 * np.pi * sigma2) + np.sum(np.log(eigenvalues + delta)) + n)

        if ll > best_ll:
            best_ll = ll

    return -2 * best_ll


def _farmcpu_burger_fastlmm(
    phe: np.ndarray,
    CV: Optional[np.ndarray],
    GK: np.ndarray
) -> float:
    """Calculate -2 * log-likelihood using FaST-LMM method.

    Uses SVD of the SNP matrix directly instead of computing kinship.

    Args:
        phe: Phenotype matrix (n × 2)
        CV: Covariates (n × c) or None
        GK: Pseudo-QTN genotypes (n × s)

    Returns:
        -2 * REML log-likelihood
    """
    y = phe[:, 1].astype(np.float64)
    n = len(y)

    # Build covariate matrix
    if CV is not None:
        X = np.column_stack([np.ones(n), CV])
    else:
        X = np.ones((n, 1))

    # Check for zero-variance columns
    if np.any(np.var(GK, axis=0) == 0):
        # Degenerate case
        return np.inf

    # SVD of SNP matrix
    U, s, Vt = np.linalg.svd(GK, full_matrices=False)

    # Keep only significant singular values
    tol = 1e-8
    k = np.sum(s > tol)
    if k == 0:
        return np.inf

    U1 = U[:, :k]
    d = s[:k]**2

    # Projections
    U1t_X = U1.T @ X
    U1t_y = U1.T @ y

    # Complement projection
    IU = np.eye(n) - U1 @ U1.T
    IU_X = IU @ X
    IU_y = IU @ y

    # Grid search for delta
    best_ll = -np.inf
    for log_delta in np.arange(-5, 5.1, 0.1):
        delta = np.exp(log_delta)

        # Compute beta
        # Part 1: from eigenspace
        beta1 = np.zeros((X.shape[1], X.shape[1]))
        beta3 = np.zeros(X.shape[1])
        for i in range(k):
            x_i = U1t_X[i:i+1, :]
            beta1 += x_i.T @ x_i / (d[i] + delta)
            beta3 += x_i.flatten() * U1t_y[i] / (d[i] + delta)

        # Part 2: from complement space
        beta2 = IU_X.T @ IU_X / delta
        beta4 = IU_X.T @ IU_y / delta

        try:
            beta = np.linalg.solve(beta1 + beta2, beta3 + beta4)
        except np.linalg.LinAlgError:
            continue

        # Log-likelihood
        # Part 1: log determinant
        log_det = np.sum(np.log(d + delta)) + (n - k) * np.log(delta)

        # Part 2: residual sum of squares
        rss1 = np.sum((U1t_y - U1t_X @ beta)**2 / (d + delta))
        rss2 = np.sum((IU_y - IU_X @ beta)**2) / delta

        sigma2 = (rss1 + rss2) / n

        ll = -0.5 * (n * np.log(2 * np.pi) + log_det + n * np.log(sigma2) + n)

        if ll > best_ll:
            best_ll = ll

    return -2 * best_ll


def PANICLE_FarmCPU(phe: np.ndarray,
               geno: Union[GenotypeMatrix, np.ndarray],
               map_data: GenotypeMap,
               CV: Optional[np.ndarray] = None,
               maxLoop: int = 10,
               p_threshold: Optional[float] = None,
               QTN_threshold: float = 0.01,
               n_eff: Optional[int] = None,
               converge: float = 1.0,
               bin_size: Optional[List[int]] = None,
               method_bin: str = "static",
               maxLine: int = 5000,
               cpu: int = 1,
               reward_method: str = "reward",
               verbose: bool = True) -> AssociationResults:
    """FarmCPU method for GWAS analysis.

    Fixed and random model Circulating Probability Unification iteratively:
    1. Uses GLM to identify candidate QTNs
    2. Bins markers and selects representative QTNs
    3. Uses GLM with selected QTNs as covariates
    4. Repeats until convergence

    Args:
        phe: Phenotype matrix (n_individuals × 2), columns [ID, trait_value]
        geno: Genotype matrix (n_individuals × n_markers)
        map_data: Genetic map with SNP positions
        CV: Covariate matrix (n_individuals × n_covariates), optional
        maxLoop: Maximum number of iterations
        p_threshold: P-value threshold for significance in first iteration.
                     If min(p) > threshold after iteration 1, stop early.
        QTN_threshold: P-value threshold for selecting pseudo-QTNs
        n_eff: Effective number of independent tests. If provided, thresholds are
               corrected by n_eff instead of n_markers.
        converge: Convergence threshold (0.0-1.0). Jaccard overlap required between
                  consecutive QTN sets to declare convergence. Default 1.0 = exact match.
                  Lower values (e.g., 0.8) allow earlier convergence.
        bin_size: List of bin sizes for iterations (default: [5e5, 5e6, 5e7])
        method_bin: Binning method ["static", "EMMA", "FaST-LMM"]
        maxLine: Batch size for GLM processing
        cpu: Number of CPUs (unused, for API compatibility)
        reward_method: How to substitute pseudo-QTN p-values ("reward", "penalty",
                       "mean", "median")
        verbose: Print progress information

    Returns:
        AssociationResults with effect sizes, standard errors, and p-values
        for all markers.
    """
    # Validate inputs
    n_individuals = phe.shape[0]
    n_markers = map_data.n_markers

    if isinstance(geno, GenotypeMatrix):
        if geno.n_individuals != n_individuals:
            raise ValueError(f"Phenotype has {n_individuals} individuals but genotype has {geno.n_individuals}")
        if geno.n_markers != n_markers:
            raise ValueError(f"Map has {n_markers} markers but genotype has {geno.n_markers}")
    else:
        if geno.shape[0] != n_individuals:
            raise ValueError(f"Phenotype has {n_individuals} individuals but genotype has {geno.shape[0]}")
        if geno.shape[1] != n_markers:
            raise ValueError(f"Map has {n_markers} markers but genotype has {geno.shape[1]}")

    # Check for missing phenotype values
    if np.any(np.isnan(phe[:, 1].astype(float))):
        raise ValueError("NAs are not allowed in phenotype")

    # Validate genotype input without loading entire matrix into memory
    # GLM handles batched imputation efficiently via get_batch_imputed()
    geno = _validate_genotype(geno, verbose=verbose)


    # Set default bin sizes (R defaults: c(5e5, 5e6, 5e7))
    if bin_size is None:
        bin_size = [int(5e5), int(5e6), int(5e7)]

    bin_selections = list(range(10, 101, 10))

    # Validate and filter covariates
    if CV is not None:
        CV = np.asarray(CV, dtype=np.float64)
        if CV.shape[0] != n_individuals:
            raise ValueError("Number of individuals doesn't match between phenotype and covariates")
        if np.any(np.isnan(CV)):
            raise ValueError("NAs are not allowed in covariates")
        # Remove constant columns
        var_mask = np.var(CV, axis=0) > 0
        CV = CV[:, var_mask]
        if CV.shape[1] == 0:
            CV = None
        npc = CV.shape[1] if CV is not None else 0
    else:
        npc = 0

    # Set QTN threshold (R: QTN.threshold = max(p.threshold, QTN.threshold))
    if p_threshold is not None:
        QTN_threshold = max(p_threshold, QTN_threshold)

    # Default p_threshold for early stopping check (R uses 0.01/nm if p.threshold is NA)
    n_tests = n_eff if n_eff is not None else n_markers
    default_p_threshold = 0.01 / n_tests

    # Compute t-critical thresholds for intermediate iterations
    # These allow us to use |t|-statistics instead of p-values for efficiency
    t_crit_early_stop = _pvalue_to_t_critical(
        p_threshold if p_threshold is not None else default_p_threshold
    )
    t_crit_qtn = _pvalue_to_t_critical(QTN_threshold)

    # Initialize iteration variables
    the_loop = 0
    seq_qtn_save: np.ndarray = np.array([], dtype=int)  # QTNs from previous iteration
    seq_qtn_pre: np.ndarray = np.array([], dtype=int)   # QTNs from 2 iterations ago
    is_done = False
    # P stores |t|-statistics for intermediate iterations (larger = more significant)
    # Converted to p-values only at the end for final output
    P: Optional[np.ndarray] = None

    # Track historical statistics for pseudo-QTNs (for substitution)
    # Key: marker index, Value: statistic from iteration before it became a covariate
    # We store the LAST valid values before the QTN was added as covariate
    qtn_pvalue_history: Dict[int, List[float]] = {}
    qtn_effect_history: Dict[int, float] = {}  # Effect from before becoming QTN
    qtn_se_history: Dict[int, float] = {}      # SE from before becoming QTN

    # Results storage
    final_effects: Optional[np.ndarray] = None
    final_se: Optional[np.ndarray] = None
    final_pvalues: Optional[np.ndarray] = None
    final_df_full: Optional[int] = None
    the_cv = CV  # Current covariates (updated each iteration)

    # Cache first iteration results to avoid redundant GLM call on early stop
    first_iter_effects: Optional[np.ndarray] = None
    first_iter_se: Optional[np.ndarray] = None
    first_iter_pvalues: Optional[np.ndarray] = None

    # Track previous iteration's effects/SE for storing QTN history
    prev_effects: Optional[np.ndarray] = None
    prev_se: Optional[np.ndarray] = None

    # Track last iteration's covariate summaries for final substitution
    final_cov_effect_summary: Optional[np.ndarray] = None
    final_cov_se_summary: Optional[np.ndarray] = None

    while not is_done:
        the_loop += 1
        if verbose:
            print(f"\nCurrent loop: {the_loop} out of maximum of {maxLoop}")

        # Step 1: Set prior (just use P from previous iteration)
        my_prior = P  # In R, FarmCPU.Prior returns P unchanged when Prior=NULL

        # Step 2: Select pseudo-QTNs via binning
        if my_prior is not None:
            # Use appropriate covariates for binning
            bin_cv = CV if the_loop <= 2 else the_cv

            seq_qtn = _farmcpu_bin(
                phe=phe,
                geno=geno,
                map_data=map_data,
                CV=bin_cv,
                P=my_prior,
                method=method_bin,
                bin_sizes=bin_size,
                bin_selections=bin_selections,
                the_loop=the_loop,
                bound=None,
                verbose=verbose,
                use_t_stats=True  # my_prior contains |t|-statistics
            )

            if verbose:
                print(f"Selected {len(seq_qtn)} pseudo-QTNs from binning")
        else:
            seq_qtn = np.array([], dtype=int)

        # Step 3: Early stopping check (R: theLoop==2 check)
        # Now uses t-statistics: max(|t|) < t_critical means no significant SNPs
        if the_loop == 2:
            if my_prior is not None:
                max_t = np.nanmax(my_prior)  # my_prior contains |t|-statistics
                if verbose:
                    # Convert to p-value for display
                    equiv_p = _t_stat_to_pvalue(np.array([max_t]))[0]
                    threshold_display = p_threshold if p_threshold is not None else default_p_threshold
                    print(f"Max |t| from previous iteration: {max_t:.2f} (equiv p={equiv_p:.2e}), threshold t={t_crit_early_stop:.2f}")
                if max_t < t_crit_early_stop:
                    seq_qtn = np.array([], dtype=int)
                    if verbose:
                        print("Top SNPs have little effect, set seqQTN to NULL!")

            # If no QTNs after early stopping check, return first iteration's GLM results
            if len(seq_qtn) == 0:
                if verbose:
                    print("No significant pseudo-QTNs found, returning cached GLM results")
                # Use cached first iteration results (avoid redundant GLM call)
                # first_iter_pvalues contains t-statistics, will be converted at end of function
                final_effects = first_iter_effects
                final_se = first_iter_se
                final_pvalues = first_iter_pvalues  # Keep as t-stats, convert at end
                break

        # Step 4: Force include previous QTNs (R logic)
        if len(seq_qtn_save) > 0 and len(seq_qtn) > 0:
            seq_qtn = np.union1d(seq_qtn, seq_qtn_save)
            if verbose:
                print(f"After forcing previous QTNs: {len(seq_qtn)} total")

        # Step 5: Filter QTNs by t-statistic threshold (R: theLoop != 1)
        # Uses t-statistics: keep QTNs where |t| > t_crit_qtn
        if the_loop >= 2 and my_prior is not None and len(seq_qtn) > 0:
            # Handle 2D arrays (if return_cov_stats=True was used)
            if my_prior.ndim == 2:
                # Filter based on the marker's own |t| (column 0)
                seq_qtn_t = my_prior[seq_qtn, 0]
            else:
                seq_qtn_t = my_prior[seq_qtn]

            if the_loop == 2:
                # Strict filtering in loop 2
                seq_qtn = np.asarray(seq_qtn).ravel()
                seq_qtn_t = np.asarray(seq_qtn_t).ravel()
                min_len = min(seq_qtn.size, seq_qtn_t.size)
                seq_qtn = seq_qtn[:min_len]
                seq_qtn_t = seq_qtn_t[:min_len]
                keep_mask = seq_qtn_t > t_crit_qtn  # |t| > threshold
            else:
                # Keep previous QTNs plus new ones above threshold
                seq_qtn = np.asarray(seq_qtn).ravel()
                seq_qtn_t = np.asarray(seq_qtn_t).ravel()
                min_len = min(seq_qtn.size, seq_qtn_t.size)
                seq_qtn = seq_qtn[:min_len]
                seq_qtn_t = seq_qtn_t[:min_len]
                keep_mask = (seq_qtn_t > t_crit_qtn) | np.isin(seq_qtn, seq_qtn_save)

            keep_mask = np.asarray(keep_mask).ravel()
            keep_mask = keep_mask[:seq_qtn.size]
            seq_qtn = seq_qtn[keep_mask]
            seq_qtn_t = seq_qtn_t[keep_mask]

            # Remove NaN entries
            valid_mask = ~np.isnan(seq_qtn_t)
            seq_qtn = seq_qtn[valid_mask]
            seq_qtn_t = seq_qtn_t[valid_mask]

            if verbose:
                print(f"After |t| filtering (threshold={t_crit_qtn:.2f}, equiv p={QTN_threshold}): {len(seq_qtn)} QTNs")
        else:
            if my_prior is not None and len(seq_qtn) > 0:
                if my_prior.ndim == 2:
                    seq_qtn_t = my_prior[seq_qtn, 0]
                else:
                    seq_qtn_t = my_prior[seq_qtn]
            else:
                seq_qtn_t = np.array([])

        # Step 6: Remove correlated QTNs (LD pruning)
        if len(seq_qtn) > 0:
            bin_geno, seq_qtn = _farmcpu_remove(
                geno=geno,
                map_data=map_data,
                seq_qtn=seq_qtn,
                seq_qtn_p=seq_qtn_t if len(seq_qtn_t) == len(seq_qtn) else (my_prior[seq_qtn] if my_prior is not None else np.zeros(len(seq_qtn))),
                threshold=0.7,
                use_t_stats=True  # seq_qtn_t contains |t|-statistics
            )
            if verbose:
                print(f"After LD pruning: {len(seq_qtn)} QTNs")
        else:
            bin_geno = None

        # Step 7: Check convergence (Jaccard similarity)
        if len(seq_qtn) > 0 and len(seq_qtn_save) > 0:
            intersection = len(np.intersect1d(seq_qtn, seq_qtn_save))
            union = len(np.union1d(seq_qtn, seq_qtn_save))
            the_converge = intersection / union if union > 0 else 0
        else:
            the_converge = 0

        # Check for cycling (same QTNs as 2 iterations ago)
        if len(seq_qtn) > 0 and len(seq_qtn_pre) > 0:
            is_circle = (len(np.union1d(seq_qtn, seq_qtn_pre)) ==
                        len(np.intersect1d(seq_qtn, seq_qtn_pre)))
        else:
            is_circle = False

        if verbose:
            if len(seq_qtn) > 0:
                print(f"seqQTN: {seq_qtn[:10]}{'...' if len(seq_qtn) > 10 else ''}")
            else:
                print("seqQTN: NULL")
            print(f"Convergence: {the_converge:.2f}")

        # Check if done
        is_done = (the_loop >= maxLoop) or (the_converge >= converge) or is_circle

        if verbose and the_loop == maxLoop:
            print(f"Total number of possible QTNs in the model: {len(seq_qtn)}")

        # Update history
        seq_qtn_pre = seq_qtn_save.copy()
        seq_qtn_save = seq_qtn.copy()

        # Step 8: Build covariates including pseudo-QTNs
        if bin_geno is not None and len(seq_qtn) > 0:
            if CV is not None:
                the_cv = np.column_stack([CV, bin_geno])
            else:
                the_cv = bin_geno.astype(np.float64)
            if verbose:
                print(f"Number of covariates in current loop: {the_cv.shape[1]}")
        else:
            the_cv = CV

        # Degrees of freedom for marker tests in this iteration's GLM
        n_fixed = 1 + (the_cv.shape[1] if the_cv is not None else 0)
        df_full = int(n_individuals - n_fixed - 1)
        if df_full <= 0:
            raise ValueError("Degrees of freedom must be positive; check covariates/QTNs")

        # Step 9: Run GLM with updated covariates
        # Use cov_pvalue_agg for memory-efficient aggregation when QTNs exist
        # This avoids creating a huge (n_markers × n_covariates) 2D array
        # Use return_t_stats=True to skip erfc computation for intermediate iterations
        has_qtns = len(seq_qtn) > 0
        agg_method = reward_method if reward_method in {"reward", "penalty", "mean"} else None
        use_cov_pvalue_agg = has_qtns and agg_method is not None
        if verbose:
            print("Running GLM...")

        glm_result = PANICLE_GLM(
            phe=phe,
            geno=geno,
            CV=the_cv,
            maxLine=maxLine,
            cpu=cpu,
            verbose=verbose,
            missing_fill_value=MISSING_FILL_VALUE,
            return_cov_stats=has_qtns and not use_cov_pvalue_agg,
            cov_pvalue_agg=agg_method if use_cov_pvalue_agg else None,
            return_t_stats=True  # Return |t|-statistics instead of p-values
        )

        effects = glm_result.effects.copy()
        se = glm_result.se.copy()
        # P now contains |t|-statistics (larger = more significant)
        if glm_result.pvalues.ndim == 2:
            P = glm_result.pvalues[:, 0].copy()
        else:
            P = glm_result.pvalues.copy()

        # Get aggregated covariate statistics for substitution (if available)
        cov_pvalue_summary = getattr(glm_result, "cov_pvalue_summary", None)
        cov_effect_summary = getattr(glm_result, "cov_effect_summary", None)
        cov_se_summary = getattr(glm_result, "cov_se_summary", None)

        # Cache first iteration results for potential early stop (avoid redundant GLM)
        if the_loop == 1:
            first_iter_effects = effects.copy()
            first_iter_se = se.copy()
            first_iter_pvalues = P.copy()  # Contains |t|-statistics

        # Step 10: Track statistics for pseudo-QTNs BEFORE they become covariates
        # Store effect/SE from PREVIOUS iteration (before becoming covariate)
        # These values are valid; after becoming a QTN, the marker's effect is collinear
        if my_prior is not None and prev_effects is not None:
            for qtn_idx in seq_qtn:
                if qtn_idx not in qtn_pvalue_history:
                    # First time seeing this QTN - record its stats from PREVIOUS iteration
                    qtn_pvalue_history[qtn_idx] = [my_prior[qtn_idx]]
                    # Store effect and SE from before it became a covariate
                    qtn_effect_history[qtn_idx] = prev_effects[qtn_idx]
                    qtn_se_history[qtn_idx] = prev_se[qtn_idx]
                # Always add current |t| (even though it's confounded when QTN is covariate)
                qtn_pvalue_history[qtn_idx].append(P[qtn_idx])

        # Save current effects/SE for next iteration's history tracking
        prev_effects = effects.copy()
        prev_se = se.copy()

        # Step 11: Substitute pseudo-QTN t-statistics using covariate statistics
        # When using return_t_stats=True, cov_pvalue_summary already contains t-stats
        # Aggregation mode: use cov_pvalue_summary for each covariate column.
        # Full mode: fall back to per-marker covariate t-stats (from pvalues array).
        if has_qtns:
            n_user_cov = CV.shape[1] if CV is not None else 0
            if cov_pvalue_summary is not None:
                # Layout: [intercept, user_cov1, ..., user_covN, QTN_1, QTN_2, ...]
                # cov_pvalue_summary already contains |t|-statistics (no conversion needed)
                qtn_start_idx = 1 + n_user_cov
                for i, qtn_idx in enumerate(seq_qtn):
                    cov_idx = qtn_start_idx + i
                    if cov_idx < len(cov_pvalue_summary):
                        sub_t = cov_pvalue_summary[cov_idx]
                        if np.isnan(sub_t) or not np.isfinite(sub_t):
                            sub_t = 0.0  # Invalid = least significant
                        P[qtn_idx] = sub_t
            elif glm_result.pvalues.ndim == 2:
                # Full covariate stats path (needed for median)
                # glm_result.pvalues contains |t|-statistics when return_t_stats=True
                qtn_start_col = 2 + n_user_cov
                for i, qtn_idx in enumerate(seq_qtn):
                    col_idx = qtn_start_col + i
                    if col_idx < glm_result.pvalues.shape[1]:
                        cov_t_vec = glm_result.pvalues[:, col_idx]
                        # For t-stats: reward=max, penalty=min (opposite of p-values)
                        if reward_method == "reward":
                            sub_t = np.nanmax(cov_t_vec)
                        elif reward_method == "penalty":
                            sub_t = np.nanmin(cov_t_vec)
                        elif reward_method == "mean":
                            sub_t = np.nanmean(cov_t_vec)
                        elif reward_method == "median":
                            sub_t = np.nanmedian(cov_t_vec)
                        else:
                            sub_t = np.nanmax(cov_t_vec)

                        if np.isnan(sub_t) or not np.isfinite(sub_t):
                            sub_t = 0.0
                        P[qtn_idx] = sub_t

        # Explicitly delete glm_result to free memory before next iteration
        del glm_result

        final_effects = effects
        final_se = se
        final_pvalues = P  # Contains |t|-statistics until converted at end
        final_df_full = df_full
        # Keep the last iteration's covariate summaries for final substitution
        final_cov_effect_summary = cov_effect_summary
        final_cov_se_summary = cov_se_summary

    # Convert t-statistics back to p-values for final output
    final_tstats = final_pvalues
    final_pvalues = _t_stat_to_pvalue(final_tstats)
    if seq_qtn_save.size > 0 and final_df_full is not None and final_df_full > 0:
        qtn_indices = seq_qtn_save.astype(int, copy=False)
        qtn_t = np.abs(final_tstats[qtn_indices])
        final_pvalues[qtn_indices] = 2.0 * stats.t.sf(qtn_t, final_df_full)

    # Substitute effects and SE for pseudo-QTNs
    # Use the covariate effect from the final model (mean across all marker tests)
    # This gives the QTN's conditional effect accounting for all other covariates
    n_user_cov = CV.shape[1] if CV is not None else 0
    qtn_start_idx = 1 + n_user_cov  # [intercept, user_covs..., QTN_1, ...]

    for i, qtn_idx in enumerate(seq_qtn_save):
        cov_idx = qtn_start_idx + i
        # Prefer aggregated covariate effect from final model (accounts for all covariates)
        if final_cov_effect_summary is not None and cov_idx < len(final_cov_effect_summary):
            effect_val = final_cov_effect_summary[cov_idx]
            if np.isfinite(effect_val):
                final_effects[qtn_idx] = effect_val
            elif qtn_idx in qtn_effect_history:
                # Fallback to pre-covariate value
                final_effects[qtn_idx] = qtn_effect_history[qtn_idx]
        elif qtn_idx in qtn_effect_history:
            final_effects[qtn_idx] = qtn_effect_history[qtn_idx]

        if final_cov_se_summary is not None and cov_idx < len(final_cov_se_summary):
            se_val = final_cov_se_summary[cov_idx]
            if np.isfinite(se_val):
                final_se[qtn_idx] = se_val
            elif qtn_idx in qtn_se_history:
                final_se[qtn_idx] = qtn_se_history[qtn_idx]
        elif qtn_idx in qtn_se_history:
            final_se[qtn_idx] = qtn_se_history[qtn_idx]

    # Create results object
    result = AssociationResults(
        effects=final_effects,
        se=final_se,
        pvalues=final_pvalues,
        snp_map=map_data
    )

    PANICLE_FarmCPU.last_selected_qtns = [int(idx) for idx in seq_qtn_save]

    if verbose:
        n_sig = np.sum(final_pvalues < 0.05 / n_markers)
        print(f"\nFarmCPU complete. {n_sig} markers below Bonferroni threshold")

    return result


PANICLE_FarmCPU.last_selected_qtns = []  # type: ignore[attr-defined]
