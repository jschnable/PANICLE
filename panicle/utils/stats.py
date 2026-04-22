"""
Statistical utilities for GWAS analysis
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import stats

QQ_LAMBDA_MAX_SAMPLE_SIZE = 200000
QQ_LAMBDA_RANDOM_SEED = 0

def bonferroni_correction(pvalues: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """Apply Bonferroni correction for multiple testing
    
    Args:
        pvalues: Array of p-values
        alpha: Family-wise error rate (default: 0.05)
        
    Returns:
        Tuple of (corrected_pvalues, corrected_threshold)
    """
    n_tests = len(pvalues)
    corrected_threshold = alpha / n_tests
    corrected_pvalues = np.minimum(pvalues * n_tests, 1.0)
    
    return corrected_pvalues, corrected_threshold

def fdr_correction(pvalues: np.ndarray, alpha: float = 0.05, method: str = 'bh') -> Tuple[np.ndarray, np.ndarray]:
    """Apply False Discovery Rate correction (Benjamini-Hochberg)
    
    Args:
        pvalues: Array of p-values
        alpha: False discovery rate (default: 0.05)
        method: Method ('bh' for Benjamini-Hochberg)
        
    Returns:
        Tuple of (rejected_hypotheses, corrected_pvalues)
    """
    pvalues = np.asarray(pvalues)
    pvalues_sortind = np.argsort(pvalues)
    pvalues_sorted = pvalues[pvalues_sortind]
    sortrevind = pvalues_sortind.argsort()
    
    if method == 'bh':
        # Benjamini-Hochberg procedure
        n = len(pvalues)
        i = np.arange(1, n + 1)
        corrected = pvalues_sorted * n / i
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        corrected_pvalues = corrected[sortrevind]
        rejected = corrected_pvalues <= alpha
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return rejected, corrected_pvalues

def compute_mac_keep_indices(
    genotype,
    min_mac: int,
    *,
    max_dosage: float = 2.0,
) -> Optional[np.ndarray]:
    """Return integer indices of markers whose minor allele count >= min_mac.

    Works on a GenotypeMatrix (preferred, uses zero-copy view) or numpy array.
    Returns None when min_mac <= 0 (filter disabled). For a diploid 0/1/2 matrix
    this is a single vectorized column-sum pass (~30 ms per 100k markers at
    1000 individuals; see /tmp/bench_mac.py).
    """
    if min_mac is None or int(min_mac) <= 0:
        return None
    if hasattr(genotype, "n_individuals") and hasattr(genotype, "n_markers"):
        n_ind = genotype.n_individuals
        n_mrk = genotype.n_markers
        if n_mrk == 0:
            return np.zeros(0, dtype=np.int64)
        arr = genotype.to_numpy(copy=False)
    else:
        arr = np.asarray(genotype)
        if arr.ndim != 2:
            raise ValueError("genotype must be 2D (n_individuals x n_markers)")
        n_ind = arr.shape[0]
        n_mrk = arr.shape[1]
        if n_mrk == 0:
            return np.zeros(0, dtype=np.int64)
    col_sums = arr.sum(axis=0, dtype=np.int64)
    max_total = int(round(max_dosage * n_ind))
    mac = np.minimum(col_sums, max_total - col_sums)
    return np.flatnonzero(mac >= int(min_mac)).astype(np.int64, copy=False)


def pad_association_results(
    result,
    keep_indices: Optional[np.ndarray],
    full_n_markers: int,
    full_map=None,
):
    """Expand a filtered AssociationResults back to full-map length with NaN fill.

    No-op when keep_indices is None (no filter applied) or the result already
    spans full_n_markers. Mirrors BLINK's internal pad-back pattern.

    Silently passes through objects that don't expose the (effects, se,
    pvalues) trio — some test mocks only implement ``.pvalues`` or
    ``.to_numpy()``, and we have no way to reconstruct a padded result for them.
    """
    if result is None:
        return result
    if not all(hasattr(result, attr) for attr in ("effects", "se", "pvalues")):
        return result

    if full_map is not None:
        if hasattr(full_map, "n_markers"):
            map_n = int(full_map.n_markers)
        elif hasattr(full_map, "to_dataframe"):
            map_n = len(full_map.to_dataframe())
        else:
            map_n = len(full_map)
        if map_n != full_n_markers:
            raise ValueError(
                f"full_map has {map_n} markers but full_n_markers is {full_n_markers}"
            )

    if keep_indices is None:
        if len(result.effects) != full_n_markers:
            raise ValueError(
                f"Unfiltered result length {len(result.effects)} does not match "
                f"full_n_markers {full_n_markers}"
            )
        if full_map is not None and hasattr(result, "snp_map"):
            result.snp_map = full_map
        return result

    if len(result.effects) == full_n_markers:
        if full_map is not None and hasattr(result, "snp_map"):
            result.snp_map = full_map
        return result
    from .data_types import AssociationResults  # local import to avoid cycle
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    if keep_indices.ndim != 1:
        raise ValueError("keep_indices must be a 1D array")
    if len(result.effects) != keep_indices.size:
        raise ValueError(
            f"Result length {len(result.effects)} does not match keep_indices length {keep_indices.size}"
        )
    if keep_indices.size and (keep_indices.min() < 0 or keep_indices.max() >= full_n_markers):
        raise IndexError("keep_indices are out of bounds for the full marker set")
    effects = np.full(full_n_markers, np.nan, dtype=float)
    se = np.full(full_n_markers, np.nan, dtype=float)
    pvalues = np.full(full_n_markers, np.nan, dtype=float)
    effects[keep_indices] = np.asarray(result.effects, dtype=float)
    se[keep_indices] = np.asarray(result.se, dtype=float)
    pvalues[keep_indices] = np.asarray(result.pvalues, dtype=float)
    return AssociationResults(
        effects=effects, se=se, pvalues=pvalues,
        snp_map=full_map,
        metadata=getattr(result, "metadata", None),
    )


def calculate_maf_from_genotypes(
    genotypes: np.ndarray,
    *,
    missing_value: int = -9,
    max_dosage: float = 2.0,
) -> np.ndarray:
    """Calculate minor allele frequencies from genotype matrix (vectorized)

    Args:
        genotypes: Genotype matrix (individuals × markers)
        missing_value: Value representing missing data
        max_dosage: Maximum genotype dosage used when converting genotype means
            into allele frequencies (default 2.0 for diploids)

    Returns:
        Array of minor allele frequencies for each marker
    """
    # Fast path for pre-imputed GenotypeMatrix (no missing values).
    if hasattr(genotypes, 'calculate_allele_frequencies') and getattr(genotypes, 'is_imputed', False):
        allele_freq = genotypes.calculate_allele_frequencies(max_dosage=max_dosage)
        maf = np.minimum(allele_freq, 1.0 - allele_freq)
        return np.asarray(maf)

    # Handle GenotypeMatrix wrapper if present (uses _data internally)
    if hasattr(genotypes, 'to_numpy'):
        genotypes = genotypes.to_numpy(copy=False)
    elif hasattr(genotypes, '_data'):
        genotypes = genotypes._data
    elif hasattr(genotypes, 'data'):
        genotypes = genotypes.data

    # Ensure we have a numpy array (handles memmap too)
    if not isinstance(genotypes, np.ndarray):
        genotypes = np.asarray(genotypes)

    # Create mask for valid (non-missing) values
    # Handle both integer and float arrays (isnan only works on floats)
    valid_mask = genotypes != missing_value
    if np.issubdtype(genotypes.dtype, np.floating):
        valid_mask = valid_mask & (~np.isnan(genotypes))

    # Use masked array for efficient computation
    masked_geno = np.ma.array(genotypes, mask=~valid_mask)

    # Calculate mean per marker (column) - vectorized
    allele_freq = masked_geno.mean(axis=0).filled(0.0) / max(max_dosage, 1e-12)

    # MAF is minimum of freq and 1-freq
    maf = np.minimum(allele_freq, 1.0 - allele_freq)

    return np.asarray(maf)

def genomic_inflation_factor(pvalues: np.ndarray) -> float:
    """Calculate genomic inflation factor (lambda).

    Uses a fast closed-form inverse for df=1: chi2.isf(p) = 2 * erfcinv(p)^2.
    """
    valid_pvals = pvalues[np.isfinite(pvalues) & (pvalues > 0) & (pvalues <= 1)]
    if len(valid_pvals) == 0:
        return 1.0

    # Avoid scipy.stats.chi2.ppf overhead for df=1
    from scipy.special import erfcinv
    clipped = np.clip(valid_pvals, 1e-300, 1.0)
    chi2_values = 2.0 * (erfcinv(clipped) ** 2)
    median_chi2 = np.median(chi2_values)
    expected_median = 0.454936423119572  # chi2.ppf(0.5, df=1)

    lambda_gc = median_chi2 / expected_median
    return lambda_gc

def qq_compatible_genomic_inflation_factor(
    pvalues: np.ndarray,
    *,
    max_sample_size: int = QQ_LAMBDA_MAX_SAMPLE_SIZE,
    random_seed: int = QQ_LAMBDA_RANDOM_SEED,
) -> Tuple[float, bool]:
    """Calculate lambda using the same subsampling strategy as QQ plotting.

    Returns:
        Tuple of (lambda_gc, is_approximate)
    """
    valid_pvals = pvalues[np.isfinite(pvalues) & (pvalues > 0) & (pvalues <= 1)]
    n_valid = len(valid_pvals)
    if n_valid == 0:
        return 1.0, False

    is_approximate = n_valid > max_sample_size
    if is_approximate:
        rng = np.random.default_rng(random_seed)
        valid_pvals = rng.choice(valid_pvals, size=max_sample_size, replace=False)

    return genomic_inflation_factor(valid_pvals), is_approximate

def qq_plot_data(pvalues: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for Q-Q plot
    
    Args:
        pvalues: Array of observed p-values
        
    Returns:
        Tuple of (expected_pvalues, observed_pvalues) for plotting
    """
    valid_pvals = pvalues[np.isfinite(pvalues) & (pvalues > 0)]
    valid_pvals = np.sort(valid_pvals)
    n = len(valid_pvals)
    
    if n == 0:
        return np.array([]), np.array([])
    
    # Expected p-values under null hypothesis
    expected_pvals = np.arange(1, n + 1) / (n + 1)
    
    return expected_pvals, valid_pvals
