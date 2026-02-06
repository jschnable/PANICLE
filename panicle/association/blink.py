"""BLINK (Bayesian-information and Linkage-disequilibrium Iteratively Nested Keyway)

Python implementation aligned with the reference rMVP/BLINK algorithm.

The current implementation focuses on parity with the R routines located in
``BLINK/R`` while leveraging PANICLE utilities for genotype handling and
high-performance GLM scanning.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

from ..utils.data_types import AssociationResults, GenotypeMap, GenotypeMatrix
from ..utils.stats import calculate_maf_from_genotypes
from .glm import PANICLE_GLM


MISSING_FILL_VALUE = 1.0  # rMVP / BLINK fill value for missing markers (-9/NA)


@dataclass
class BlinkIterationDetails:
    """Diagnostic information for a single BLINK iteration."""

    iteration: int
    n_candidates: int
    n_ld_pruned: int
    n_selected: int
    overlap_with_previous: float
    min_pvalue: float
    threshold_used: float
    selected_qtns: Tuple[int, ...]


def PANICLE_BLINK(
    phe: np.ndarray,
    geno: Union[GenotypeMatrix, np.ndarray],
    map_data: GenotypeMap,
    CV: Optional[np.ndarray] = None,
    Prior: Optional[Union[np.ndarray, "pd.DataFrame"]] = None,
    *,
    maxLoop: int = 10,
    converge: float = 1.0,
    ld_threshold: float = 0.7,
    maf_threshold: float = 0.0,
    bic_method: str = "naive",
    method_sub: str = "reward",
    p_threshold: Optional[float] = None,
    qtn_threshold: float = 0.01,
    cut_off: float = 0.01,
    fdr_cut: bool = False,
    maxLine: int = 5000,
    cpu: int = 1,
    max_genotype_dosage: float = 2.0,
    verbose: bool = True,
) -> AssociationResults:
    """Run BLINK GWAS analysis.

    Args:
        phe: Phenotype matrix (n_individuals × 2), columns [ID, trait_value]
        geno: Genotype matrix (n_individuals × n_markers) or GenotypeMatrix
        map_data: Genetic map describing SNPs
        CV: Optional covariates (n_individuals × n_covariates)
        Prior: Optional prior information (SNP, Chr, Pos, weight)
        maxLoop: Maximum BLINK iterations
        converge: Jaccard similarity threshold to declare convergence
        ld_threshold: r^2 threshold for LD pruning
        maf_threshold: Minimum MAF for markers to be retained
        bic_method: Strategy for BIC candidate evaluation (naive/even/lg/ln/fixed)
        method_sub: Substitution rule for pseudo-QTN statistics
        p_threshold: Optional global p-value cutoff (iteration 2 logic)
        qtn_threshold: Optional ceiling on pseudo QTN selection threshold (applied to iterations > 2)
        cut_off: Significance level used for Bonferroni/FDR thresholding in iteration 2
        fdr_cut: If True, use FDR-based cutoff for iteration 2 (matches GAPIT Blink)
        maxLine: Batch size for PANICLE_GLM streaming
        cpu: Number of CPU cores (forwarded to PANICLE_GLM)
        max_genotype_dosage: Maximum genotype dosage used when computing allele
            frequencies/MAF (default 2.0 for diploids)
        verbose: Print progress information

    Returns:
        AssociationResults aligned to the original map order.
    """

    if phe.ndim != 2 or phe.shape[1] != 2:
        raise ValueError("Phenotype matrix must have shape (n, 2) with [ID, Trait]")

    trait_values = phe[:, 1].astype(np.float64)
    if np.isnan(trait_values).any():
        raise ValueError("BLINK currently requires complete trait observations")

    genotype_array, major_alleles = _ensure_numpy_genotype(geno)
    geno_is_imputed = isinstance(geno, GenotypeMatrix) and geno.is_imputed
    n_individuals, n_markers_total = genotype_array.shape

    if trait_values.shape[0] != n_individuals:
        raise ValueError("Phenotype and genotype sample sizes do not match")

    cv_matrix = _prepare_covariates(CV, n_individuals)
    map_df = map_data.to_dataframe().reset_index(drop=True)

    maf_mask, maf_values = _compute_maf_mask(
        genotype_array,
        maf_threshold,
        max_genotype_dosage,
    )
    filtered_indices = np.where(maf_mask)[0]
    if len(filtered_indices) == 0:
        raise ValueError("All markers were removed by the MAF threshold")

    geno_filtered = genotype_array[:, filtered_indices]
    major_filtered = (
        major_alleles[filtered_indices] if major_alleles is not None else None
    )
    glm_geno: Union[GenotypeMatrix, np.ndarray] = geno_filtered
    if geno_is_imputed:
        glm_geno = GenotypeMatrix(
            geno_filtered,
            precompute_alleles=False,
            is_imputed=True,
        )
    map_filtered = map_df.loc[filtered_indices].reset_index(drop=True)
    chrom_values, pos_values = _precompute_map_coordinates(map_filtered)
    map_filtered_map = GenotypeMap(map_filtered)

    # Apply a gentle warning when markers were trimmed for transparency
    if len(filtered_indices) != n_markers_total and verbose:
        n_removed = n_markers_total - len(filtered_indices)
        warnings.warn(
            f"BLINK: removed {n_removed} markers below MAF threshold {maf_threshold:.4f}",
            RuntimeWarning,
        )

    iteration_details: List[BlinkIterationDetails] = []
    qtn_history: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    iteration_qtn_log: List[Dict[str, Union[int, List[int], float]]] = []
    selected_qtns_prev: List[int] = []
    selected_qtns_last: List[int] = []
    selected_qtns_for_union: List[int] = []
    pvalues_initial: Optional[np.ndarray] = None

    current_covariates = cv_matrix
    current_pvalues = np.ones(len(filtered_indices))

    for iteration in range(1, maxLoop + 1):
        if verbose:
            print(f"BLINK iteration {iteration}/{maxLoop}")

        prev_selected = list(selected_qtns_for_union)

        if iteration == 1:
            seq_qtns: List[int] = []
        else:
            pvals_for_selection = current_pvalues.copy()
            # Apply corrections if there were previously selected QTNs
            if prev_selected:
                if pvalues_initial is not None:
                    idx = np.asarray(prev_selected, dtype=int)
                    pvals_for_selection[idx] = pvalues_initial[idx]

            # Apply prior information using FarmCPU logic for shared behavior
            if Prior is not None:
                prior_adjusted = _farmcpu_prior_simple(
                    genetic_map=map_filtered,
                    pvalues=pvals_for_selection,
                    prior_info=Prior,
                    kinship_algorithm="FARM-CPU",
                )
                if prior_adjusted is not None:
                    pvals_for_selection = prior_adjusted

            # Select and refine QTNs (this should happen for all iterations > 1)
            size_for_threshold = pvals_for_selection.size
            seq_qtns, meta = _select_and_refine_qtns(
                pvalues=pvals_for_selection,
                iteration=iteration,
                qtn_threshold=qtn_threshold,
                cut_off=cut_off,
                fdr_cut=fdr_cut,
                p_threshold=p_threshold,
                previous_qtns=prev_selected,
                genotype=geno_filtered,
                map_df=map_filtered,
                chrom_values=chrom_values,
                pos_values=pos_values,
                map_data=map_filtered_map,
                ld_threshold=ld_threshold,
                bic_method=bic_method,
                trait_values=trait_values,
                base_covariates=cv_matrix,
                fill_value=MISSING_FILL_VALUE,
                verbose=verbose,
            )


            if iteration == 2 and meta.get("min_pvalue") is not None:
                min_prior = meta.get("min_pvalue", float("nan"))
                threshold_used = meta.get("threshold", cut_off / max(size_for_threshold, 1))
                if np.isfinite(min_prior) and min_prior > threshold_used:
                    if verbose:
                        print("Top snps have little effect, set seqQTN to NULL!")
                    seq_qtns = []

            if iteration == 2 and len(seq_qtns) == 0:
                if verbose:
                    print("No pseudo-QTNs identified at iteration 2. Stopping BLINK early.")
                break

            if verbose:
                print("  Previous QTNs before union:", prev_selected)
            if prev_selected:
                seq_qtns = sorted(set(seq_qtns).union(prev_selected))
                if verbose:
                    print(
                        "  Union with previous QTNs:",
                        "current=", seq_qtns,
                        "previous=", prev_selected,
                    )

            if iteration > 2 and len(seq_qtns) > 1:
                bic_selected, _ = _bic_model_selection(
                    geno_filtered,
                    seq_qtns,
                    trait_values,
                    cv_matrix,
                    bic_method=bic_method,
                    fill_value=MISSING_FILL_VALUE,
                    verbose=verbose,
                )

                prev_union = [idx for idx in seq_qtns if idx in prev_selected]
                if verbose:
                    print(
                        "  BIC merged selection check:",
                        "current=", seq_qtns,
                        "previous=", prev_selected,
                        "bic=", bic_selected,
                    )
                if prev_union:
                    merged_ordered: List[int] = []
                    seen = set()
                    for idx in seq_qtns:
                        if idx in bic_selected or idx in prev_union:
                            if idx not in seen:
                                merged_ordered.append(idx)
                                seen.add(idx)
                    bic_selected = merged_ordered

                if not bic_selected:
                    bic_selected = seq_qtns
                    if verbose:
                        print("  BIC fallback: retaining union of QTNs", bic_selected)

                seq_qtns = bic_selected

            current_covariates = _assemble_covariates(
                base_covariates=cv_matrix,
                genotype=geno_filtered,
                qtn_indices=seq_qtns,
                fill_value=MISSING_FILL_VALUE,
                major_alleles=major_filtered,
            )
            # DEBUG: Log covariate matrix details
            if verbose:
                n_base = cv_matrix.shape[1] if cv_matrix is not None else 0
                n_qtns = len(seq_qtns)
                print(f"  Assembled covariates: {n_base} base + {n_qtns} QTNs = {current_covariates.shape[1] if current_covariates is not None else 0} total")
        
        glm_results = PANICLE_GLM(
            phe=phe,
            geno=glm_geno,
            CV=current_covariates,
            maxLine=maxLine,
            cpu=cpu,
            verbose=verbose and iteration == 1,
            impute_missing=True,
            major_alleles=major_filtered,
            missing_fill_value=MISSING_FILL_VALUE,
        )
        result_array = glm_results.to_numpy()
        raw_effects = result_array[:, 0].copy()
        raw_se = result_array[:, 1].copy()
        raw_pvalues = result_array[:, 2].copy()

        # Apply GAPIT-style p-value post-processing
        # R: P[P==0] <- min(P[P!=0],na.rm=TRUE)*0.01
        # R: P[is.na(P)] =1
        zero_mask = raw_pvalues == 0
        if np.any(zero_mask):
            nonzero_pvals = raw_pvalues[raw_pvalues > 0]
            if len(nonzero_pvals) > 0:
                min_nonzero = np.min(nonzero_pvals)
                raw_pvalues[zero_mask] = min_nonzero * 0.01

        # Set NaN p-values to 1
        nan_mask = np.isnan(raw_pvalues)
        if np.any(nan_mask):
            raw_pvalues[nan_mask] = 1.0

        # Update result array with post-processed p-values
        result_array[:, 2] = raw_pvalues

        if iteration == 1:
            pvalues_initial = raw_pvalues.copy()
            qtn_history.clear()
        elif seq_qtns:
            for stale_idx in list(qtn_history.keys()):
                if stale_idx not in seq_qtns:
                    del qtn_history[stale_idx]

            for marker_idx in seq_qtns:
                eff = float(raw_effects[marker_idx])
                se = float(raw_se[marker_idx])
                pval = float(raw_pvalues[marker_idx])
                qtn_history[marker_idx].append({
                    "iteration": iteration,
                    "p": pval,
                    "effect": eff,
                    "se": se,
                })
                if verbose:
                    print(f"  QTN {marker_idx} GLM stats: p={pval:.6e}, effect={eff:.4f}")

            active_history = {
                idx: qtn_history[idx]
                for idx in seq_qtns
                if qtn_history[idx]
            }
            if active_history:
                _apply_substitution(
                    result_array,
                    active_history,
                    method_sub=method_sub,
                    initial_pvalues=pvalues_initial,
                )
        else:
            qtn_history.clear()

        current_pvalues = result_array[:, 2].astype(np.float64, copy=True)
        min_pvalue = float(np.nanmin(current_pvalues)) if current_pvalues.size else np.nan

        overlap = _jaccard_similarity(seq_qtns, prev_selected)
        threshold_used = (
            float(meta.get("threshold", np.nan)) if iteration > 1 else
            _determine_iteration_threshold(iteration, len(filtered_indices), p_threshold)
        )
        iteration_details.append(
            BlinkIterationDetails(
                iteration=iteration,
                n_candidates=int(meta["n_candidates"]) if iteration > 1 else len(filtered_indices),
                n_ld_pruned=int(meta["n_after_ld"]) if iteration > 1 else len(filtered_indices),
                n_selected=len(seq_qtns),
                overlap_with_previous=overlap,
                min_pvalue=min_pvalue,
                threshold_used=threshold_used,
                selected_qtns=tuple(seq_qtns),
            )
        )

        iteration_qtn_log.append({
            "iteration": iteration,
            "selected_qtns": list(seq_qtns),
            "overlap": overlap,
        })

        if verbose:
            print("  Iteration completed with QTNs:", seq_qtns)

        selected_qtns_for_union = list(seq_qtns)

        if iteration > 1 and overlap >= converge:
            if verbose:
                print(
                    "BLINK convergence reached:"
                    f" overlap={overlap:.3f} (>= {converge})"
                )
            selected_qtns_last = list(seq_qtns)
            selected_qtns_for_union = list(seq_qtns)
            break

    selected_qtns_prev = list(selected_qtns_last)
    selected_qtns_last = list(seq_qtns)

    if selected_qtns_last and current_covariates is not None:
        cov_p, cov_effects, cov_se = _get_covariate_statistics(phe, current_covariates, verbose=False)
        if cov_p.size:
            n_base_cov = current_covariates.shape[1] - len(selected_qtns_last)
            for cov_offset, qtn_idx in enumerate(selected_qtns_last):
                cov_idx = n_base_cov + cov_offset
                if cov_idx >= len(cov_p):
                    continue
                qtn_history[qtn_idx].append({
                    "iteration": "final",
                    "p": float(cov_p[cov_idx]),
                    "effect": float(cov_effects[cov_idx]),
                    "se": float(cov_se[cov_idx]),
                })

    final_history: Dict[int, List[Dict[str, float]]] = {}
    if selected_qtns_last:
        final_history = {
            idx: qtn_history[idx]
            for idx in selected_qtns_last
            if idx in qtn_history and qtn_history[idx]
        }

    if final_history:
        _apply_substitution(
            result_array,
            final_history,
            method_sub=method_sub,
            initial_pvalues=pvalues_initial,
        )

    effects_full = np.full(n_markers_total, np.nan, dtype=np.float64)
    se_full = np.full(n_markers_total, np.nan, dtype=np.float64)
    pvalues_full = np.full(n_markers_total, np.nan, dtype=np.float64)

    effects_full[filtered_indices] = result_array[:, 0]
    se_full[filtered_indices] = result_array[:, 1]
    pvalues_full[filtered_indices] = result_array[:, 2]

    association = AssociationResults(
        effects=effects_full,
        se=se_full,
        pvalues=pvalues_full,
        snp_map=map_data,
    )

    PANICLE_BLINK.last_iteration_details = iteration_details
    PANICLE_BLINK.last_selected_qtns = selected_qtns_last
    PANICLE_BLINK.last_iteration_qtns = iteration_qtn_log

    return association


def _ensure_numpy_genotype(
    geno: Union[GenotypeMatrix, np.ndarray]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(geno, GenotypeMatrix):
        genotype_array = np.asarray(geno._data)  # type: ignore[attr-defined]
        major_alleles = geno.major_alleles
    else:
        genotype_array = np.asarray(geno)
        major_alleles = None
    if genotype_array.ndim != 2:
        raise ValueError("Genotype matrix must be 2-dimensional")
    return genotype_array, major_alleles


def _precompute_map_coordinates(map_df: "pd.DataFrame") -> Tuple[np.ndarray, np.ndarray]:
    """Precompute chromosome/position arrays for fast candidate ordering."""

    import pandas as pd  # noqa: F401 - preserved for type checking parity

    chrom_series = map_df["CHROM"]
    if np.issubdtype(chrom_series.dtype, np.number):
        chrom_values = chrom_series.to_numpy(dtype=float, copy=False)
    else:
        chrom_as_str = chrom_series.astype(str)
        chrom_order = sorted(chrom_as_str.unique(), key=lambda val: (len(val), val))
        chrom_lookup = {val: float(idx) for idx, val in enumerate(chrom_order)}
        chrom_values = chrom_as_str.map(chrom_lookup).to_numpy(dtype=float)

    if "POS" in map_df.columns:
        pos_values = map_df["POS"].to_numpy(dtype=float, copy=False)
    elif "Position" in map_df.columns:
        pos_values = map_df["Position"].to_numpy(dtype=float, copy=False)
    else:
        pos_values = np.arange(map_df.shape[0], dtype=float)

    return chrom_values, pos_values


def _prepare_covariates(CV: Optional[np.ndarray], n_individuals: int) -> Optional[np.ndarray]:
    if CV is None:
        return None
    cv_array = np.asarray(CV)
    if cv_array.ndim == 1:
        cv_array = cv_array.reshape(-1, 1)
    if cv_array.shape[0] != n_individuals:
        raise ValueError("Covariate rows must equal number of individuals")
    return cv_array.astype(np.float64, copy=False)


def _compute_maf_mask(
    genotype: np.ndarray,
    maf_threshold: float,
    max_genotype_dosage: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if maf_threshold <= 0.0:
        maf = calculate_maf_from_genotypes(
            genotype,
            max_dosage=max_genotype_dosage,
        )
        mask = np.ones_like(maf, dtype=bool)
        return mask, maf
    maf = calculate_maf_from_genotypes(
        genotype,
        max_dosage=max_genotype_dosage,
    )
    mask = maf >= maf_threshold
    return mask, maf


def remove_qtns_by_ld(
    selected_qtns: Sequence[int],
    genotype_matrix: Union[np.ndarray, GenotypeMatrix],
    correlation_threshold: float = 0.7,
    *,
    within_chrom_only: bool = True,
    map_data: Optional[GenotypeMap] = None,
    verbose: bool = False,
    debug: bool = False,
    ld_max: Optional[int] = None,
    max_samples: int = 100000,
) -> List[int]:
    if not selected_qtns:
        return []

    seq_qtn = np.asarray(selected_qtns, dtype=int)

    if map_data is not None:
        chrom_values = map_data.chromosomes.values
        pos_values = map_data.positions.values
        seen = set()
        unique_qtns: List[int] = []
        for idx in seq_qtn:
            key = (chrom_values[idx], pos_values[idx])
            if key in seen:
                continue
            seen.add(key)
            unique_qtns.append(int(idx))
        seq_qtn = np.asarray(unique_qtns, dtype=int)

    if seq_qtn.size == 0:
        return []

    if within_chrom_only and map_data is not None:
        chrom_values = map_data.chromosomes.values
        chrom_subset = chrom_values[seq_qtn]
        kept_set: set[int] = set()
        for chrom in np.unique(chrom_subset):
            chrom_mask = chrom_subset == chrom
            group = seq_qtn[chrom_mask]
            kept_group = _ld_prune_subset(
                group,
                genotype_matrix=genotype_matrix,
                correlation_threshold=correlation_threshold,
                max_samples=max_samples,
            )
            kept_set.update(int(idx) for idx in kept_group)
        kept = [int(idx) for idx in seq_qtn if int(idx) in kept_set]
    else:
        kept = [
            int(idx)
            for idx in _ld_prune_subset(
                seq_qtn,
                genotype_matrix=genotype_matrix,
                correlation_threshold=correlation_threshold,
                max_samples=max_samples,
            )
        ]

    if ld_max is not None:
        kept = kept[: max(ld_max, 0)]

    if verbose and debug:
        print(f"  LD pruning retained {len(kept)} QTNs")

    return kept


def _ld_prune_subset(
    seq_qtn: np.ndarray,
    *,
    genotype_matrix: Union[np.ndarray, GenotypeMatrix],
    correlation_threshold: float,
    max_samples: int,
) -> np.ndarray:
    if seq_qtn.size <= 1:
        return seq_qtn

    if isinstance(genotype_matrix, GenotypeMatrix):
        x = genotype_matrix.get_columns_imputed(seq_qtn, dtype=np.float32)
    else:
        x = genotype_matrix[:, seq_qtn].astype(np.float32)
        missing = (x == -9) | np.isnan(x)
        if missing.any():
            x[missing] = MISSING_FILL_VALUE

    if x.shape[0] > max_samples:
        x = x[:max_samples, :]

    if x.shape[1] <= 1:
        return seq_qtn

    std = np.std(x, axis=0)
    constant_cols = std < 1e-10
    std[constant_cols] = 1.0
    x_centered = (x - np.mean(x, axis=0)) / std
    x_centered[:, constant_cols] = 0.0

    r = np.corrcoef(x_centered.T)
    if r.ndim == 0:
        r = np.array([[r]])
    r = np.nan_to_num(r, nan=0.0)

    keep = np.ones(len(seq_qtn), dtype=bool)
    for i in range(len(seq_qtn)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(seq_qtn)):
            if keep[j] and abs(r[i, j]) > correlation_threshold:
                keep[j] = False

    return seq_qtn[keep]


def _get_covariate_statistics(
    phe: np.ndarray,
    covariates: Optional[np.ndarray],
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if covariates is None or covariates.size == 0:
        return np.array([]), np.array([]), np.array([])

    cov = np.asarray(covariates, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)

    y = np.asarray(phe[:, 1], dtype=np.float64)
    n = y.shape[0]
    if cov.shape[0] != n:
        raise ValueError("Covariate rows must equal number of individuals")

    X = np.column_stack([np.ones(n, dtype=np.float64), cov])
    n_cov = cov.shape[1]
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1] or n <= X.shape[1]:
        return np.ones(n_cov), np.zeros(n_cov), np.ones(n_cov)

    # Suppress spurious FPE warnings from Accelerate/BLAS matmul on arm64
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX, rcond=1e-12)

        beta = XtX_inv @ (X.T @ y)
        residuals = y - X @ beta
    df = n - X.shape[1]
    if df <= 0:
        return np.ones(n_cov), np.zeros(n_cov), np.ones(n_cov)

    rss = float(np.sum(residuals ** 2))
    if rss <= 0:
        rss = np.finfo(np.float64).tiny
    sigma2 = rss / df

    var_beta = np.diag(XtX_inv) * sigma2
    se_beta = np.sqrt(np.maximum(var_beta, 1e-30))
    t_stats = np.zeros_like(beta)
    valid = se_beta > 0
    if np.any(valid):
        t_stats[valid] = beta[valid] / se_beta[valid]
    pvals = np.ones_like(beta)
    if np.any(valid):
        pvals[valid] = 2.0 * stats.t.sf(np.abs(t_stats[valid]), df)

    cov_effects = beta[1:].astype(np.float64, copy=False)
    cov_se = se_beta[1:].astype(np.float64, copy=False)
    cov_pvals = pvals[1:].astype(np.float64, copy=False)

    bad = ~np.isfinite(cov_effects) | ~np.isfinite(cov_se) | ~np.isfinite(cov_pvals)
    if bad.any():
        cov_effects = cov_effects.copy()
        cov_se = cov_se.copy()
        cov_pvals = cov_pvals.copy()
        cov_effects[bad] = 0.0
        cov_se[bad] = 1.0
        cov_pvals[bad] = 1.0

    if verbose:
        print(f"Computed covariate stats for {n_cov} covariates")

    return cov_pvals, cov_effects, cov_se


def _farmcpu_prior_simple(
    genetic_map: "pd.DataFrame",
    pvalues: Optional[np.ndarray] = None,
    prior_info: Optional[Union[np.ndarray, "pd.DataFrame"]] = None,
    kinship_algorithm: str = "FARM-CPU"
) -> Optional[np.ndarray]:
    """
    Apply prior information matching R reference FarmCPU.Prior logic.

    Args:
        genetic_map: Genetic map DataFrame with SNP column
        pvalues: P-values array to modify
        prior_info: Prior information with SNP names and weights
        kinship_algorithm: Algorithm name (used for early returns)

    Returns:
        Modified p-values or None if no modification needed
    """
    # R reference early returns
    if prior_info is None and kinship_algorithm != "FARM-CPU":
        return pvalues
    if prior_info is None and pvalues is None:
        return pvalues
    if pvalues is None:
        return None

    # Apply prior weights if provided
    if prior_info is not None:
        try:
            import pandas as pd
        except ImportError:
            warnings.warn("pandas required for prior processing")
            return pvalues

        if isinstance(prior_info, pd.DataFrame):
            prior_df = prior_info
        else:
            prior_df = pd.DataFrame(prior_info)

        if prior_df.shape[1] < 4:
            return pvalues

        # Ensure proper column names
        prior_df.columns = ["SNP", "Chr", "Pos", "Weight"] + list(prior_df.columns[4:])

        # R reference: index=match(Prior[,1],GM[,1],nomatch = 0)
        updated_pvalues = pvalues.copy()

        for _, prior_row in prior_df.iterrows():
            prior_snp = prior_row["SNP"]
            weight = float(prior_row["Weight"])

            # Find matching SNP in genetic map - R uses nomatch=0 (no match)
            matches = genetic_map["SNP"] == prior_snp
            if matches.any():
                matching_indices = np.where(matches)[0]
                for idx in matching_indices:
                    if idx < len(updated_pvalues):
                        # R reference: P[index]=P[index]*Prior[,4]
                        updated_pvalues[idx] *= weight

        return updated_pvalues

def _select_and_refine_qtns(
    *,
    pvalues: np.ndarray,
    iteration: int,
    qtn_threshold: float,
    cut_off: float,
    fdr_cut: bool,
    p_threshold: Optional[float],
    previous_qtns: Sequence[int],
    genotype: np.ndarray,
    map_df: "pd.DataFrame",
    chrom_values: np.ndarray,
    pos_values: np.ndarray,
    map_data: GenotypeMap,
    ld_threshold: float,
    bic_method: str,
    trait_values: np.ndarray,
    base_covariates: Optional[np.ndarray],
    fill_value: float,
    verbose: bool,
) -> Tuple[List[int], Dict[str, float]]:
    finite_mask = np.isfinite(pvalues)
    candidate_indices = np.where(finite_mask)[0]
    pvals_finite = pvalues[candidate_indices]

    n_markers = len(pvalues)


    if iteration == 2:
        if p_threshold is not None and not np.isnan(p_threshold):
            threshold_val = float(p_threshold)
        else:
            bonf_cutoff = cut_off / max(n_markers, 1)
            threshold_val = bonf_cutoff
            if fdr_cut and pvals_finite.size > 0:
                # Exact GAPIT logic reproduction
                # R: sp = sort(seqQTN.p)
                # R: spd = abs(cutOff - sp * nm/cutOff)
                # R: index_fdr = grep(min(spd), spd)[1]
                # R: FDRcutoff = cutOff * index_fdr/nm
                sp = np.sort(pvals_finite)
                spd = np.abs(cut_off - sp * (n_markers / cut_off))

                # Find first occurrence of minimum (R's grep behavior)
                min_val = np.min(spd)
                index_fdr = np.where(spd == min_val)[0][0] + 1  # 1-based like R
                fdr_cutoff = cut_off * index_fdr / max(n_markers, 1)
                threshold_val = min(threshold_val, fdr_cutoff)
    else:
        base_threshold = _determine_iteration_threshold(iteration, n_markers, p_threshold)
        threshold_val = base_threshold
        if iteration >= 3:
            threshold_val = min(threshold_val, qtn_threshold)

    keep_mask = pvals_finite < threshold_val
    initial_candidates = candidate_indices[keep_mask]

    if initial_candidates.size == 0:
        return [], {
            "n_candidates": 0,
            "n_after_ld": 0,
            "threshold": float(threshold_val),
            "min_pvalue": float(np.nanmin(pvalues)) if np.any(finite_mask) else float('nan'),
        }

    pvals_subset = pvalues[initial_candidates]
    chrom_subset = chrom_values[initial_candidates]
    pos_subset = pos_values[initial_candidates]
    order = np.lexsort((pos_subset, chrom_subset, pvals_subset))
    ordered_candidates = initial_candidates[order].tolist()

    # Delegate LD pruning to FarmCPU's validated Blink-style LD removal
    ld_pruned = remove_qtns_by_ld(
        selected_qtns=ordered_candidates,
        genotype_matrix=genotype,
        correlation_threshold=ld_threshold,
        map_data=map_data,
        within_chrom_only=True,
        verbose=verbose and iteration == 2,
        debug=False,
        ld_max=None,
    )

    if not ld_pruned:
        return [], {
            "n_candidates": len(initial_candidates),
            "n_after_ld": 0,
            "threshold": float(threshold_val),
            "min_pvalue": float(np.nanmin(pvalues)) if np.any(finite_mask) else float('nan'),
        }

    if verbose:
        print(
            "  LD survivors:",
            " ".join(str(idx) for idx in ld_pruned),
        )

    selected, _ = _bic_model_selection(
        genotype,
        ld_pruned,
        trait_values,
        base_covariates,
        bic_method=bic_method,
        fill_value=fill_value,
        iteration=iteration,
        verbose=verbose,
    )

    prev_in_ld: List[int] = [idx for idx in ld_pruned if idx in previous_qtns]
    if prev_in_ld:
        selected_set = set(selected)
        if verbose:
            print(
                "  Preserving prior QTNs after BIC:",
                " ".join(str(idx) for idx in prev_in_ld),
            )
        merged: List[int] = []
        seen = set()
        for idx in ld_pruned:
            if idx in selected_set or idx in prev_in_ld:
                if idx not in seen:
                    merged.append(idx)
                    seen.add(idx)
        selected = merged

    if not selected and ld_pruned:
        selected = [ld_pruned[0]]
        if verbose:
            print(
                "  BIC rejected all QTNs, falling back to strongest candidate",
                ld_pruned[0],
            )

    return selected, {
        "n_candidates": len(initial_candidates),
        "n_after_ld": len(ld_pruned),
        "threshold": float(threshold_val),
        "min_pvalue": float(np.nanmin(pvalues)) if np.any(finite_mask) else float('nan'),
    }


def _determine_iteration_threshold(
    iteration: int,
    n_markers: int,
    p_threshold: Optional[float],
) -> float:
    if p_threshold is not None and not np.isnan(p_threshold):
        return float(p_threshold)
    if iteration == 2:
        return 0.01 / max(n_markers, 1)
    return 1.0 / max(n_markers, 1)


def _assemble_covariates(
    *,
    base_covariates: Optional[np.ndarray],
    genotype: np.ndarray,
    qtn_indices: Sequence[int],
    fill_value: float,
    major_alleles: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if not qtn_indices:
        return base_covariates

    qtn_matrix = genotype[:, qtn_indices].astype(np.float64, copy=True)
    missing_mask = (qtn_matrix == -9) | np.isnan(qtn_matrix)
    if missing_mask.any():
        if major_alleles is not None:
            fill_values = major_alleles[qtn_indices].astype(np.float64)
            qtn_matrix[missing_mask] = np.broadcast_to(fill_values, qtn_matrix.shape)[missing_mask]
        else:
            qtn_matrix[missing_mask] = float(fill_value)

    if base_covariates is None:
        return qtn_matrix
    return np.column_stack([base_covariates, qtn_matrix])


def _bic_model_selection(
    genotype: np.ndarray,
    candidate_indices: Sequence[int],
    trait_values: np.ndarray,
    base_covariates: Optional[np.ndarray],
    *,
    bic_method: str,
    fill_value: float,
    iteration: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[int], np.ndarray]:
    """
    Improved BIC model selection following GAPIT Blink.BICselection exactly
    """
    if not candidate_indices:
        return [], np.array([])

    n = trait_values.shape[0]
    threshold = max(int(np.floor(n / np.log(max(n, 2)))), 1)

    candidate_list = [int(idx) for idx in candidate_indices[:threshold]]
    if not candidate_list:
        return [], np.array([])

    m = len(candidate_list)
    positions = _gapit_bic_positions(m, bic_method) or [m]

    BICv = np.full(len(positions), np.nan)
    pmatrix = np.ones((m, m))
    bic_trace: List[Tuple[int, float, List[int]]] = []

    geno_matrix = genotype[:, candidate_list].astype(np.float64, copy=True)
    miss_mask = (geno_matrix == -9) | np.isnan(geno_matrix)
    if miss_mask.any():
        geno_matrix[miss_mask] = float(fill_value)

    for idx, pos in enumerate(positions):
        if pos > m:
            pos = m

        geno_subset = geno_matrix[:, :pos]
        design = _build_design_matrix(
            trait_values,
            base_covariates,
            geno_subset,
            fill_value,
        )

        if design is None or design.shape[0] <= design.shape[1]:
            BICv[idx] = np.inf
            if verbose:
                print(f"    Skipping position {pos}: singular design matrix")
            continue

        bic_val, stats_matrix = _compute_bic_statistics(
            design,
            trait_values,
            iteration=iteration,
            pos=pos,
            verbose=verbose,
        )
        BICv[idx] = bic_val

        if stats_matrix.size:
            qtn_stats = stats_matrix[-pos:]
            pmatrix[:pos, pos-1] = qtn_stats[:, 2]

        if verbose:
            bic_trace.append((pos, bic_val, candidate_list[:pos].copy()))

    valid_bic = BICv[~np.isnan(BICv)]
    if len(valid_bic) == 0:
        return [], np.array([])

    min_bic_idx = np.nanargmin(BICv)
    best_pos = positions[min_bic_idx]

    selected_qtns = candidate_list[:best_pos]
    if verbose and bic_trace:
        print("  BIC evaluation summary (pos, bic, qtns):")
        for pos, bic_val, qtns in bic_trace:
            marker = "*" if pos == best_pos else "-"
            qtn_str = " ".join(str(idx) for idx in qtns)
            print(f"    {marker} pos={pos:<3d} bic={bic_val:>14.3f} qtns={qtn_str}")
        print(f"  Raw BIC values: {BICv}")

    if len(selected_qtns) > 0:
        selected_pvalues = pmatrix[:len(selected_qtns), len(selected_qtns)-1]

        # Construct statistics array (effects, se, pvalues, t_stats)
        # Use the final model statistics
        best_stats = np.column_stack([
            np.zeros(len(selected_qtns)),      # effects - will be calculated in main GLM
            np.ones(len(selected_qtns)),       # se - will be calculated in main GLM
            selected_pvalues,                  # p-values from BIC selection
            np.zeros(len(selected_qtns))       # t-stats - will be calculated in main GLM
        ])
    else:
        best_stats = np.array([])

    return selected_qtns, best_stats


def _gapit_bic_positions(m: int, method: str) -> List[int]:
    """
    GAPIT Blink.BICselection position calculation - exact implementation
    """
    if m <= 0:
        return []

    if method == "naive":
        return list(range(1, m + 1))
    elif method == "even":
        step_length = int(np.sqrt(m)) + 1
        step = int(m / step_length)
        if (m - step * step_length) >= (0.5 * step_length):
            step = step + 1
        if step_length > m:
            step_length = m
            step = 1
        position = list(range(step, m + 1, step))
        if position[-1] < m:
            position.append(m)
        return position
    elif method == "lg":
        if m == 1:
            return [1]
        else:
            le = np.arange(1, m + 1, dtype=float)
            step = le / np.log10(le)
            step[0] = 1  # Handle log10(1) = 0
            for i in range(1, m):
                le[i] = le[i-1] + step[i]
                le = np.round(le).astype(int)
                if le[i] > m:
                    return le[:i+1].tolist()
            return le.tolist()
    elif method == "ln":
        if m == 1:
            return [1]
        else:
            le = np.arange(1, m + 1, dtype=float)
            step = le / np.log(le)
            step[0] = 1  # Handle log(1) = 0
            for i in range(1, m):
                le[i] = le[i-1] + step[i]
                le = np.round(le).astype(int)
                if le[i] > m:
                    return le[:i+1].tolist()
            return le.tolist()
    elif method == "fixed":
        if m > 20:
            return list(np.floor(np.linspace(1, m, 20)).astype(int))
        else:
            return list(range(1, m + 1))
    else:
        # Default to naive if unknown method
        return list(range(1, m + 1))

def _bic_positions(m: int, method: str) -> List[int]:
    if m <= 0:
        return []
    method = method.lower()
    if method == "naive":
        return list(range(1, m + 1))
    if method == "even":
        step_len = int(np.floor(np.sqrt(m))) + 1
        step = int(np.floor(m / step_len)) if step_len > 0 else 1
        if (m - step * step_len) >= (0.5 * step_len):
            step += 1
        if step_len > m:
            step_len = m
            step = 1
        step = max(step, 1)
        positions = list(range(step, m + 1, step))
        if not positions or positions[-1] < m:
            positions.append(m)
        return positions
    if method == "lg":
        if m == 1:
            return [1]
        le = np.arange(1, m + 1, dtype=float)
        denom = np.log10(np.maximum(le, 2))
        denom[denom == 0] = np.nan
        step = le / denom
        temp = le.copy()
        positions: List[int] = []
        for i in range(1, m):
            incr = step[i]
            if np.isnan(incr) or incr <= 0:
                incr = 1.0
            temp[i] = temp[i - 1] + incr
            temp = np.round(temp)
            if temp[i] > m:
                positions = temp[:i].astype(int).tolist()
                break
        else:
            positions = temp.astype(int).tolist()
        if not positions:
            positions = [1]
        if positions[-1] != m:
            positions.append(m)
        return sorted(set(pos for pos in positions if pos >= 1))
    if method == "ln":
        if m == 1:
            return [1]
        le = np.arange(1, m + 1, dtype=float)
        denom = np.log(np.maximum(le, 2))
        denom[denom == 0] = np.nan
        step = le / denom
        temp = le.copy()
        positions: List[int] = []
        for i in range(1, m):
            incr = step[i]
            if np.isnan(incr) or incr <= 0:
                incr = 1.0
            temp[i] = temp[i - 1] + incr
            temp = np.round(temp)
            if temp[i] > m:
                positions = temp[:i].astype(int).tolist()
                break
        else:
            positions = temp.astype(int).tolist()
        if not positions:
            positions = [1]
        if positions[-1] != m:
            positions.append(m)
        return sorted(set(pos for pos in positions if pos >= 1))
    if method == "fixed":
        if m > 20:
            positions = [int(np.floor(x)) for x in np.linspace(1, m, 20)]
            positions = [pos if pos >= 1 else 1 for pos in positions]
            positions = sorted(set(positions))
            if positions[-1] != m:
                positions.append(m)
            return positions
        return list(range(1, m + 1))
    warnings.warn(f"Unknown BIC method '{method}', defaulting to naive")
    return list(range(1, m + 1))

def _build_design_matrix(
    trait_values: np.ndarray,
    base_covariates: Optional[np.ndarray],
    genotype_subset: np.ndarray,
    fill_value: float,
) -> Optional[np.ndarray]:
    n = trait_values.shape[0]
    geno = genotype_subset.astype(np.float64, copy=True)
    missing_mask = (geno == -9) | np.isnan(geno)
    if missing_mask.any():
        geno[missing_mask] = float(fill_value)

    components = [np.ones((n, 1))]
    if base_covariates is not None and base_covariates.size > 0:
        components.append(base_covariates)
    components.append(geno)
    X = np.column_stack(components)
    if X.shape[0] <= X.shape[1]:
        return None
    return X


def _compute_bic_statistics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    iteration: Optional[int] = None,
    pos: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    n, k = X.shape
    # NOTE: Some Accelerate builds (NumPy 2.0 arm64) raise spurious FPE flags on matmul.
    # Compute under a local errstate and validate the result instead of crashing.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        XtX = X.T @ X
        if verbose:
            iter_label = f"{iteration}" if iteration is not None else "?"
            pos_label = f", pos={pos}" if pos is not None else ""
            cond_val = np.linalg.cond(XtX)
            print(f"  Iteration {iter_label}{pos_label}: cond(XtX) = {cond_val:.2e}, X.shape = {X.shape}")
        try:
            XtX_inv = np.linalg.pinv(XtX, rcond=1e-12)
        except np.linalg.LinAlgError:
            return np.inf, np.array([])

        beta = XtX_inv @ (X.T @ y)
        Xbeta = X @ beta
    if verbose:
        print(
            "  X: shape={shape}, min={min:.2e}, max={max:.2e}".format(
                shape=X.shape,
                min=np.min(X),
                max=np.max(X),
            )
        )
        if X.size > 0:
            last_col = X[:, -1]
            unique_vals = np.unique(last_col)
            if unique_vals.size <= 10:
                uniq_repr = np.array2string(unique_vals, precision=3, separator=", ")
            else:
                uniq_repr = f"{unique_vals.size} unique"
            print(
                "  X[:,-1]: min={min:.2e}, max={max:.2e}, mean={mean:.2e}, std={std:.2e}, uniq={uniq}".format(
                    min=np.min(last_col),
                    max=np.max(last_col),
                    mean=np.mean(last_col),
                    std=np.std(last_col),
                    uniq=uniq_repr,
                )
            )
        print(
            "  y: shape={shape}, min={min:.2e}, max={max:.2e}".format(
                shape=y.shape,
                min=np.min(y),
                max=np.max(y),
            )
        )
        print(
            "  beta: shape={shape}, min={min:.2e}, max={max:.2e}".format(
                shape=beta.shape,
                min=np.min(beta),
                max=np.max(beta),
            )
        )
        print(
            "  X @ beta: min={min:.2e}, max={max:.2e}".format(
                min=np.min(Xbeta),
                max=np.max(Xbeta),
            )
        )
        residuals_preview = y - Xbeta
        print(
            "  residuals would be: min={min:.2e}, max={max:.2e}".format(
                min=np.min(residuals_preview),
                max=np.max(residuals_preview),
            )
        )
    residuals = y - Xbeta
    df = n - k
    if df <= 0:
        return np.inf, np.array([])
    rss = float(np.sum(residuals ** 2))
    if rss <= 0:
        rss = np.finfo(np.float64).tiny
    sigma2 = rss / df

    bic = n * np.log(sigma2) + k * np.log(n)

    var_beta = sigma2 * np.diag(XtX_inv)
    se_beta = np.sqrt(np.maximum(var_beta, 1e-30))
    t_stats = np.zeros_like(beta)
    valid = se_beta > 0
    t_stats[valid] = beta[valid] / se_beta[valid]
    pvals = 2 * stats.t.sf(np.abs(t_stats), df)

    return bic, np.column_stack([beta, se_beta, pvals, t_stats])


def _extract_covariate_stats(
    phe: np.ndarray,
    covariates: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if covariates is None or covariates.size == 0:
        return np.array([]), np.array([]), np.array([])
    cov_p, cov_effects, cov_se = _get_covariate_statistics(phe, covariates, verbose=False)
    return cov_p, cov_effects, cov_se


def _apply_substitution_r_style(
    result_array: np.ndarray,
    selected_qtns: List[int],
    genetic_map: np.ndarray,
    qtn_history: Dict[int, List[Dict[str, float]]],
    *,
    method_sub: str,
    model: str = "A",
    n_covariates: int = 0,
) -> None:
    """
    Apply R-style substitution matching Blink.SUB logic.

    Args:
        result_array: GLM results matrix (n_markers x n_columns)
        selected_qtns: List of selected QTN indices
        genetic_map: Genetic map array with SNP names
        qtn_history: History of QTN statistics across iterations
        method_sub: Substitution method ("reward", "penalty", "mean", "median", "onsite")
        model: Model type ("A" for additive)
        n_covariates: Number of base covariates
    """
    if not result_array.size or not selected_qtns:
        return

    method = method_sub.lower()
    if method not in {"onsite", "reward", "penalty", "mean", "median"}:
        warnings.warn(f"Unknown substitution method '{method}', defaulting to reward")
        method = "reward"

    n_qtns = len(selected_qtns)
    if n_qtns == 0:
        return

    # R reference: position=match(QTN[,1], GM[,1], nomatch = 0)
    # We already have the positions as selected_qtns

    # Calculate column indices in GLM P matrix following R reference
    n_total_cols = result_array.shape[1]

    if model == "A":
        # R: index=(ncol(GLM$P)-nqtn):(ncol(GLM$P)-1)
        # R: spot=ncol(GLM$P)
        qtn_effect_columns = list(range(n_total_cols - n_qtns, n_total_cols))
        final_pvalue_column = n_total_cols - 1
    else:  # model == "AD"
        # R: index=(ncol(GLM$P)-nqtn-1):(ncol(GLM$P)-2)
        # R: spot=ncol(GLM$P)-1
        qtn_effect_columns = list(range(n_total_cols - n_qtns - 1, n_total_cols - 1))
        final_pvalue_column = n_total_cols - 2

    # Debug output matching R reference
    total_qtns = sum(len(history) for history in qtn_history.values())
    print(f"    DEBUG R-style substitution: Processing {total_qtns} history entries with method '{method}'")

    # Apply aggregation method to each QTN
    for i, qtn_idx in enumerate(selected_qtns):
        if qtn_idx not in qtn_history or not qtn_history[qtn_idx]:
            continue

        history = qtn_history[qtn_idx]

        # Apply R reference aggregation logic
        if method == "penalty":
            # R: P.QTN=apply(GLM$P[,index],2,max,na.rm=TRUE)
            aggregated_p = max(rec["p"] for rec in history)
            best_rec = max(history, key=lambda rec: rec["p"])
        elif method == "reward":
            # R: P.QTN=apply(GLM$P[,index],2,min,na.rm=TRUE)
            aggregated_p = min(rec["p"] for rec in history)
            best_rec = min(history, key=lambda rec: rec["p"])
        elif method == "mean":
            # R: P.QTN=apply(GLM$P[,index],2,mean,na.rm=TRUE)
            aggregated_p = float(np.mean([rec["p"] for rec in history]))
            best_rec = history[-1]  # Use latest for effect estimates
        elif method == "median":
            # R: P.QTN=apply(GLM$P[,index],2,stats::median,na.rm=TRUE)
            aggregated_p = float(np.median([rec["p"] for rec in history]))
            best_rec = history[-1]  # Use latest for effect estimates
        elif method == "onsite":
            # R: P.QTN=GLM$P0[(length(GLM$P0)-nqtn+1):length(GLM$P0)]
            # This would use initial p-values, but we'll use latest for simplicity
            best_rec = history[-1]
            aggregated_p = best_rec["p"]
        else:
            best_rec = min(history, key=lambda rec: rec["p"])
            aggregated_p = best_rec["p"]

        # R reference: GLM$P[position,spot]=P.QTN
        if final_pvalue_column < result_array.shape[1]:
            result_array[qtn_idx, final_pvalue_column] = aggregated_p

        # Update effect estimates if available
        # R reference: GLM$B[position,]=GLM$betapred[(no.cv+1):length(GLM$betapred)]
        # We'll use the best record's effect and se
        if result_array.shape[1] > 1:
            result_array[qtn_idx, 0] = best_rec["effect"]  # Effect
            result_array[qtn_idx, 1] = best_rec["se"]      # Standard error


def _apply_substitution(
    result_array: np.ndarray,
    qtn_history: Dict[int, List[Dict[str, float]]],
    *,
    method_sub: str,
    initial_pvalues: Optional[np.ndarray],
) -> None:
    if not qtn_history:
        return
    method = method_sub.lower()
    if method not in {"onsite", "reward", "penalty", "mean", "median"}:
        warnings.warn(f"Unknown substitution method '{method}', defaulting to reward")
        method = "reward"

    total_qtns = sum(len(history) for history in qtn_history.values())
    if total_qtns == 0:
        return

    for marker_idx, history in qtn_history.items():
        if not history:
            continue

        if method == "onsite":
            chosen = history[-1]
            p_use = float(initial_pvalues[marker_idx]) if initial_pvalues is not None else chosen["p"]
        elif method == "reward":
            chosen = min(history, key=lambda rec: rec["p"])
            p_use = chosen["p"]
        elif method == "penalty":
            chosen = max(history, key=lambda rec: rec["p"])
            p_use = chosen["p"]
        elif method == "mean":
            p_use = float(np.mean([rec["p"] for rec in history]))
            chosen = history[-1]
        elif method == "median":
            p_use = float(np.median([rec["p"] for rec in history]))
            chosen = history[-1]

        result_array[marker_idx, 0] = chosen["effect"]
        result_array[marker_idx, 1] = chosen["se"]
        result_array[marker_idx, 2] = p_use


def _aggregate_history(
    history: Sequence[Dict[str, float]],
    method: str,
    initial_pvalues: Optional[np.ndarray],
    marker_idx: int,
) -> Optional[Tuple[float, float, float]]:
    if not history:
        return None

    if method == "onsite":
        if initial_pvalues is None or marker_idx >= len(initial_pvalues):
            return None
        latest = history[-1]
        return latest["effect"], latest["se"], float(initial_pvalues[marker_idx])

    if method in {"reward", "penalty"}:
        key_func = min if method == "reward" else max
        best_entry = key_func(history, key=lambda rec: rec["p"])
        print(f"      DEBUG aggregation: {method} method selected p={best_entry['p']:.6e} from {len(history)} history entries")
        return best_entry["effect"], best_entry["se"], best_entry["p"]

    if method == "mean":
        pvals = np.array([rec["p"] for rec in history], dtype=np.float64)
        effects = np.array([rec["effect"] for rec in history], dtype=np.float64)
        ses = np.array([rec["se"] for rec in history], dtype=np.float64)
        return float(np.mean(effects)), float(np.mean(ses)), float(np.mean(pvals))

    if method == "median":
        pvals = np.array([rec["p"] for rec in history], dtype=np.float64)
        effects = np.array([rec["effect"] for rec in history], dtype=np.float64)
        ses = np.array([rec["se"] for rec in history], dtype=np.float64)
        return (
            float(np.median(effects)),
            float(np.median(ses)),
            float(np.median(pvals)),
        )

    warnings.warn(f"Unknown substitution method '{method}', defaulting to reward")
    best_entry = min(history, key=lambda rec: rec["p"])
    return best_entry["effect"], best_entry["se"], best_entry["p"]


def _jaccard_similarity(a: Sequence[int], b: Sequence[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a.union(set_b)
    if not union:
        return 1.0
    return len(set_a.intersection(set_b)) / len(union)


# Attributes exposed for debugging/test harnesses
PANICLE_BLINK.last_iteration_details = []  # type: ignore[attr-defined]
PANICLE_BLINK.last_selected_qtns: List[int] = []  # type: ignore[attr-defined]
PANICLE_BLINK.last_iteration_qtns = []  # type: ignore[attr-defined]
