"""
Statistical utilities for GWAS analysis
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import stats

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
