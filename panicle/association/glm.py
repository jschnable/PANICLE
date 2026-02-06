"""
General Linear Model (GLM) for GWAS analysis.

This module provides the `MVP_GLM` function, which uses the Frisch-Waugh-Lovell (FWL)
theorem and QR decomposition for high-performance association testing.

Implementation details:
- panicle.association.glm_fwl_qr: Fast vectorized implementation (default).
"""

from typing import Optional, Union
import numpy as np

from ..utils.data_types import GenotypeMatrix, AssociationResults
from .glm_fwl_qr import PANICLE_GLM_ultrafast

def PANICLE_GLM(phe: np.ndarray,
           geno: Union[GenotypeMatrix, np.ndarray],
           CV: Optional[np.ndarray] = None,
           maxLine: int = 5000,
           cpu: int = 1,
           verbose: bool = True,
           impute_missing: bool = True,
           major_alleles: Optional[np.ndarray] = None,
           missing_fill_value: float = 1.0,
           return_cov_stats: bool = False,
           cov_pvalue_agg: Optional[str] = None,
           return_t_stats: bool = False) -> AssociationResults:
    """General Linear Model (GLM) for GWAS.

    Uses an optimized FWL+QR algorithm for speed.

    Args:
        phe: Phenotype array (n_individuals x 2) [ID, Value]
        geno: Genotype matrix (n_individuals x n_markers)
        CV: Covariates (n_individuals x n_covariates)
        maxLine: Batch size for processing
        cpu: Unused (kept for API compatibility)
        verbose: Print progress
        impute_missing: Unused (always handled internally by FWL+QR loader)
        major_alleles: Unused (always handled internally)
        missing_fill_value: Value to use for missing genotypes (default: 1.0)
        return_cov_stats: If True, returns effects/SE/P-values for all columns
                         including covariates. Memory intensive for large datasets.
        cov_pvalue_agg: Memory-efficient alternative to return_cov_stats.
                       Computes aggregated covariate p-values per covariate column.
                       Options: "reward" (min), "penalty" (max), "mean".
                       Result has .cov_pvalue_summary attribute with shape (n_covariates,).
        return_t_stats: If True, return |t|-statistics instead of p-values.
                       Skips erfc computation for efficiency. Useful for FarmCPU
                       intermediate iterations where only ordinal ranking matters.

    Returns:
        AssociationResults object with effects, SEs, and p-values (or |t| if return_t_stats).
        If return_cov_stats is True, results arrays will be 2D (markers x terms).
        If cov_pvalue_agg is set, arrays are 1D with cov_pvalue_summary metadata.
    """
    return PANICLE_GLM_ultrafast(
        phe=phe,
        geno=geno,
        CV=CV,
        maxLine=maxLine,
        cpu=cpu,
        verbose=verbose,
        missing_fill_value=missing_fill_value,
        return_cov_stats=return_cov_stats,
        cov_pvalue_agg=cov_pvalue_agg,
        return_t_stats=return_t_stats
    )
