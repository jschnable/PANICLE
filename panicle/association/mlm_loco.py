"""
LOCO (Leave-One-Chromosome-Out) MLM wrapper.

This module is intentionally standalone so it can be removed cleanly if LOCO
is not adopted.
"""

import time
import warnings
from typing import Optional, Union, Tuple, Dict
import numpy as np

from ..utils.data_types import GenotypeMatrix, AssociationResults
from ..matrix.kinship_loco import PANICLE_K_VanRaden_LOCO, LocoKinship, _extract_chromosomes, _group_markers_by_chrom
from .mlm import PANICLE_MLM, estimate_variance_components_brent, _calculate_neg_ml_likelihood
from .lrt import fit_marker_lrt

# Check for joblib availability
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def _subset_genotypes(geno: Union[GenotypeMatrix, np.ndarray],
                      indices: np.ndarray) -> np.ndarray:
    """Return a genotype submatrix for a set of marker indices.

    For GenotypeMatrix, uses get_columns_imputed which handles -9 and NaN.
    For pre-imputed GenotypeMatrix, skips -9 checks for faster access.
    For numpy arrays, handles -9 sentinel and NaN values by imputing to 0.
    """
    if isinstance(geno, GenotypeMatrix):
        if geno.is_imputed:
            # Data is pre-imputed, skip -9 checks for faster access
            return geno._data[:, indices].astype(np.float32)
        return geno.get_columns_imputed(indices)
    # For numpy arrays, handle missing values
    subset = geno[:, indices].astype(np.float32)
    missing_mask = (subset == -9) | np.isnan(subset)
    if missing_mask.any():
        subset[missing_mask] = 0.0
    return subset


def _process_chromosome(chrom: str,
                        indices: np.ndarray,
                        geno_subset: np.ndarray,
                        phe: np.ndarray,
                        K_loco: np.ndarray,
                        eigenK: Dict[str, np.ndarray],
                        CV: Optional[np.ndarray],
                        vc_method: str,
                        maxLine: int) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process a single chromosome for LOCO MLM.

    This function is designed to be called in parallel for each chromosome.
    Takes pre-computed K_loco and eigenK to avoid serializing the full LocoKinship object.

    Returns:
        Tuple of (chrom, indices, effects, std_errors, pvalues)
    """
    res = PANICLE_MLM(
        phe=phe,
        geno=geno_subset,
        K=K_loco,
        eigenK=eigenK,
        CV=CV,
        vc_method=vc_method,
        maxLine=maxLine,
        cpu=1,  # Don't nest parallelism
        verbose=False,
    )

    return chrom, indices, res.effects, res.se, res.pvalues


def PANICLE_MLM_LOCO(phe: np.ndarray,
                 geno: Union[GenotypeMatrix, np.ndarray],
                 map_data,
                 loco_kinship: Optional[LocoKinship] = None,
                 CV: Optional[np.ndarray] = None,
                 vc_method: str = "BRENT",
                 maxLine: int = 1000,
                 cpu: int = 1,
                 lrt_refinement: bool = True,
                 screen_threshold: float = 1e-4,
                 verbose: bool = True) -> AssociationResults:
    """Run MLM with LOCO kinship matrices grouped by chromosome.

    By default, markers with promising Wald p-values (< screen_threshold) are
    re-tested using the exact Likelihood Ratio Test (LRT) for more accurate
    p-values. This provides LRT-quality results with minimal runtime overhead.

    Args:
        phe: Phenotype matrix (n_individuals × 2), columns [ID, trait_value]
        geno: Genotype matrix (n_individuals × n_markers)
        map_data: Genetic map with chromosome information
        loco_kinship: Pre-computed LOCO kinship (computed if None)
        CV: Covariate matrix (n_individuals × n_covariates), optional
        vc_method: Variance component estimation method ["BRENT"]
        maxLine: Batch size for processing markers
        cpu: Number of CPU cores for parallel chromosome processing
        lrt_refinement: Apply LRT refinement to top hits (default: True)
        screen_threshold: P-value threshold for LRT refinement (default: 1e-4)
        verbose: Print progress information

    Returns:
        AssociationResults object containing Effect, SE, and P-value for each marker
    """
    if not isinstance(phe, np.ndarray) or phe.ndim != 2 or phe.shape[1] != 2:
        raise ValueError("Phenotype must be numpy array with 2 columns [ID, trait_value]")
    trait_values_full = phe[:, 1].astype(np.float64)
    if not np.all(np.isfinite(trait_values_full)):
        raise ValueError(
            "Phenotype contains missing/non-finite values; filter individuals before PANICLE_MLM_LOCO"
        )
    if CV is not None:
        if CV.shape[0] != phe.shape[0]:
            raise ValueError("Covariate matrix must have same number of rows as phenotypes")
        try:
            CV = CV.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            raise ValueError("Covariate matrix must contain numeric values")
        if not np.all(np.isfinite(CV)):
            raise ValueError(
                "Covariate matrix contains missing/non-finite values; filter individuals before PANICLE_MLM_LOCO"
            )

    if isinstance(geno, GenotypeMatrix):
        n_markers = geno.n_markers
    elif isinstance(geno, np.ndarray):
        n_markers = geno.shape[1]
    else:
        raise ValueError("Genotype must be GenotypeMatrix or numpy array")

    chrom_values = _extract_chromosomes(map_data, n_markers)
    chrom_groups = _group_markers_by_chrom(chrom_values)

    if loco_kinship is None:
        loco_kinship = PANICLE_K_VanRaden_LOCO(geno, map_data, maxLine=maxLine, verbose=verbose)

    effects = np.zeros(n_markers, dtype=np.float64)
    std_errors = np.zeros(n_markers, dtype=np.float64)
    p_values = np.ones(n_markers, dtype=np.float64)

    # Filter out empty chromosome groups
    chrom_items = [(chrom, indices) for chrom, indices in chrom_groups.items() if indices.size > 0]
    n_chroms = len(chrom_items)

    if verbose:
        print("=" * 60)
        print("LOCO MLM")
        print("=" * 60)
        print(f"Chromosomes: {n_chroms}")

    # Handle cpu=0 to mean use all available cores
    if cpu == 0:
        import multiprocessing
        cpu = multiprocessing.cpu_count()

    # Determine if we should use parallel processing
    use_parallel = HAS_JOBLIB and cpu > 1 and n_chroms > 1

    if use_parallel:
        if verbose:
            print(f"Using parallel processing with {min(cpu, n_chroms)} workers")

        # Pre-compute all data needed for parallel processing
        # This avoids serializing the full LocoKinship object to workers
        chrom_data = []
        for chrom, indices in chrom_items:
            geno_subset = _subset_genotypes(geno, indices)
            eigenK = loco_kinship.get_eigen(chrom)
            K_loco = loco_kinship.get_loco(chrom).to_numpy()
            chrom_data.append((chrom, indices, geno_subset, K_loco, eigenK))

        # Process chromosomes in parallel
        # Use 'loky' for CPU-bound work (releases GIL in numpy/numba)
        results = Parallel(n_jobs=min(cpu, n_chroms), backend='loky')(
            delayed(_process_chromosome)(
                chrom, indices, geno_subset, phe, K_loco, eigenK, CV, vc_method, maxLine
            )
            for chrom, indices, geno_subset, K_loco, eigenK in chrom_data
        )

        # Collect results
        for chrom, indices, eff, se, pvals in results:
            effects[indices] = eff
            std_errors[indices] = se
            p_values[indices] = pvals

    else:
        # Sequential processing (original behavior)
        if verbose and not HAS_JOBLIB and cpu > 1:
            print("Note: joblib not available, using sequential processing")

        for chrom, indices in chrom_items:
            if verbose:
                print(f"Processing chromosome {chrom} ({indices.size} markers)")

            geno_subset = _subset_genotypes(geno, indices)
            eigenK = loco_kinship.get_eigen(chrom)
            K_loco = loco_kinship.get_loco(chrom)

            res = PANICLE_MLM(
                phe=phe,
                geno=geno_subset,
                K=K_loco,
                eigenK=eigenK,
                CV=CV,
                vc_method=vc_method,
                maxLine=maxLine,
                cpu=1,
                verbose=False,
            )

            effects[indices] = res.effects
            std_errors[indices] = res.se
            p_values[indices] = res.pvalues

    # -------------------------------------------------------------------------
    # LRT Refinement Phase (if enabled)
    # -------------------------------------------------------------------------
    if lrt_refinement:
        candidate_indices = np.where(p_values < screen_threshold)[0]
        n_candidates = len(candidate_indices)

        if verbose:
            print(f"LRT refinement: {n_candidates} candidates (p < {screen_threshold})")

        if n_candidates > 0:
            # Extract phenotype values and setup covariates
            trait_values = trait_values_full
            n_individuals = len(trait_values)

            if CV is not None:
                X = np.column_stack([np.ones(n_individuals), CV])
            else:
                X = np.ones((n_individuals, 1))

            # Cache for null models by chromosome
            null_cache: Dict[str, Dict[str, np.ndarray]] = {}

            def _get_null_model(chrom: str) -> Dict[str, np.ndarray]:
                if chrom in null_cache:
                    return null_cache[chrom]

                eigen = loco_kinship.get_eigen(chrom)
                eigenvals = np.maximum(
                    np.nan_to_num(np.asarray(eigen['eigenvals'], dtype=np.float64), nan=1e-6, posinf=1e6, neginf=1e-6),
                    1e-6
                )
                eigenvecs = np.nan_to_num(np.asarray(eigen['eigenvecs'], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    y_transformed = eigenvecs.T @ trait_values
                    X_transformed = eigenvecs.T @ X
                y_transformed = np.nan_to_num(y_transformed, nan=0.0, posinf=0.0, neginf=0.0)
                X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)

                delta_null, vg_null, ve_null = estimate_variance_components_brent(
                    y_transformed, X_transformed, eigenvals, verbose=False, use_ml=True
                )
                h2_null = vg_null / (vg_null + ve_null) if (vg_null + ve_null) > 0 else 0.0
                null_neg_loglik = _calculate_neg_ml_likelihood(h2_null, y_transformed, X_transformed, eigenvals)

                payload = {
                    "eigenvals": eigenvals,
                    "eigenvecs": eigenvecs,
                    "y_transformed": y_transformed,
                    "X_transformed": X_transformed,
                    "null_neg_loglik": null_neg_loglik,
                }
                null_cache[chrom] = payload
                return payload

            # Process candidates
            start_time = time.time()
            for i, marker_idx in enumerate(candidate_indices):
                if verbose and i % 10 == 0:
                    print(f"  Refining marker {i+1}/{n_candidates}...", end='\r')

                chrom = str(chrom_values[marker_idx])
                null_model = _get_null_model(chrom)

                # Get genotype
                if isinstance(geno, GenotypeMatrix):
                    g_raw = geno.get_batch_imputed(marker_idx, marker_idx+1).flatten()
                else:
                    g_raw = geno[:, marker_idx]

                g_raw = np.nan_to_num(np.asarray(g_raw, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

                # Transform genotype
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    g_transformed = null_model["eigenvecs"].T @ g_raw
                g_transformed = np.nan_to_num(g_transformed, nan=0.0, posinf=0.0, neginf=0.0)

                # Run LRT
                lrt_stat, lrt_p, lrt_beta, lrt_se = fit_marker_lrt(
                    null_model["y_transformed"],
                    null_model["X_transformed"],
                    g_transformed,
                    null_model["eigenvals"],
                    null_model["null_neg_loglik"]
                )

                # Update results if LRT model is numerically stable
                if np.isfinite(lrt_p) and np.isfinite(lrt_beta) and np.isfinite(lrt_se):
                    p_values[marker_idx] = lrt_p
                    effects[marker_idx] = lrt_beta
                    std_errors[marker_idx] = lrt_se

            duration = time.time() - start_time
            if verbose:
                print(f"  LRT refinement complete in {duration:.2f}s ({duration/max(1,n_candidates):.3f}s/marker)")

    return AssociationResults(effects=effects, se=std_errors, pvalues=p_values)
