"""
LOCO (Leave-One-Chromosome-Out) MLM wrapper.

This module is intentionally standalone so it can be removed cleanly if LOCO
is not adopted.
"""

import time
import warnings
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List

import numpy as np

from ..utils.data_types import (
    GenotypeMatrix,
    AssociationResults,
    ensure_eager_genotype,
    impute_numpy_batch_major_allele,
)
from ..matrix.kinship_loco import (
    PANICLE_K_VanRaden_LOCO,
    LocoKinship,
    _resolve_chromosome_groups,
)
from .mlm import (
    HAS_NUMBA,
    PANICLE_MLM,
    compute_fast_pvalues,
    estimate_variance_components_brent,
    estimate_variance_components_brent_batch,
    estimate_null_ml_batch,
    _calculate_neg_ml_likelihood,
    process_batch_parallel,
)
from .multi_trait_store import MultiTraitResultStore, marker_table

if HAS_NUMBA:
    import numba
from .lrt import fit_markers_lrt_batch_prebuilt
from ._validation import missing_values_error
from ..utils.perf import available_cpu_count, numba_thread_limit, scaled_thread_count

# Minimum markers per thread before a parallel numba kernel is worth its
# thread-launch overhead (small-job guard for the LRT refinement and Wald scan).
_LRT_MARKERS_PER_THREAD = 128
_WALD_MARKERS_PER_THREAD = 128

# Traits per block in the trait-batched Wald scan. The per-block temporaries
# scale as block * q0 * maxLine floats, so 1024 traits with a 1000-marker batch
# and a handful of covariates stays in the tens of MB.
_DEFAULT_TRAIT_BLOCK = 1024
# Auto block sizing keeps the per-block (traits x chromosome markers) result
# arrays (effects, se, p: 24 bytes per cell) under this many bytes.
_BLOCK_RESULT_BYTES = 1 << 30

# NOTE: LOCO MLM no longer uses joblib. Chromosomes run sequentially and
# parallelism is marker-level inside PANICLE_MLM (numba prange pinned to the
# CPU budget via numba_thread_limit).


def _subset_genotypes(
    geno: Union[GenotypeMatrix, np.ndarray], indices: np.ndarray
) -> np.ndarray:
    """Return a genotype submatrix for a set of marker indices.

    For GenotypeMatrix, uses get_columns_imputed which handles -9 and NaN.
    For pre-imputed GenotypeMatrix, skips -9 checks for faster access.
    For numpy arrays, handles -9 sentinel and NaN values by major-allele
    imputation (matching GenotypeMatrix behavior).
    """
    if isinstance(geno, GenotypeMatrix):
        if geno.is_imputed:
            # Data is pre-imputed, skip -9 checks for faster access
            return geno.get_columns(indices, dtype=np.float32)
        return geno.get_columns_imputed(indices)
    # For numpy arrays, handle missing values
    subset = geno[:, indices]
    return impute_numpy_batch_major_allele(subset, fill_value=None, dtype=np.float32)


def _build_design_matrix(n_individuals: int, CV: Optional[np.ndarray]) -> np.ndarray:
    """Build fixed-effect design matrix with intercept."""
    if CV is None:
        return np.ones((n_individuals, 1), dtype=np.float64)
    return np.column_stack([np.ones(n_individuals), CV])


def _run_wald_with_pretransformed_genotypes(
    y_transformed: np.ndarray,
    X_transformed: np.ndarray,
    eigenvals: np.ndarray,
    chrom_geno_eigen_f32: np.ndarray,
    *,
    vc_method: str,
    maxLine: int,
    cpu: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Wald tests for one trait with chromosome genotypes already in eigenspace."""
    if vc_method.upper() != "BRENT":
        raise ValueError(f"Unknown variance component method: {vc_method}")

    n_markers = chrom_geno_eigen_f32.shape[1]
    if n_markers == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.ones(0, dtype=np.float64),
        )

    delta_hat, vg_hat, _ = estimate_variance_components_brent(
        y_transformed,
        X_transformed,
        eigenvals,
        verbose=False,
    )

    y_f32 = y_transformed.astype(np.float32, copy=False)
    X_f32 = X_transformed.astype(np.float32, copy=False)

    eig_safe = np.maximum(eigenvals, 1e-6)
    V_eigenvals = eig_safe + delta_hat
    weights_f32 = (1.0 / V_eigenvals).astype(np.float32)

    XTW_f32 = X_f32.T * weights_f32
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            UXWUX_f32 = XTW_f32 @ X_f32
            UXWy_f32 = XTW_f32 @ y_f32
    if (not np.all(np.isfinite(UXWUX_f32))) or (not np.all(np.isfinite(UXWy_f32))):
        UXWUX_f32 = (XTW_f32.astype(np.float64) @ X_f32.astype(np.float64)).astype(
            np.float32
        )
        UXWy_f32 = (XTW_f32.astype(np.float64) @ y_f32.astype(np.float64)).astype(
            np.float32
        )
    UXWUX_f64 = UXWUX_f32.astype(np.float64)
    UXWy_f64 = UXWy_f32.astype(np.float64)
    wy_f32 = weights_f32 * y_f32

    effects = np.zeros(n_markers, dtype=np.float64)
    std_errors = np.zeros(n_markers, dtype=np.float64)
    p_values = np.ones(n_markers, dtype=np.float64)
    n_batches = (n_markers + maxLine - 1) // maxLine

    # Sequential batches with the per-marker numba kernels pinned to the CPU
    # budget (BLAS still parallelizes each cross-product matmul), scaled down for
    # small marker counts. Mirrors the PANICLE_MLM cleanup: no joblib-over-batches
    # layer, so this does not nest with the all-core BLAS/numba work per batch.
    with numba_thread_limit(scaled_thread_count(cpu, n_markers, _WALD_MARKERS_PER_THREAD)):
        for batch_idx in range(n_batches):
            start_marker = batch_idx * maxLine
            end_marker = min(start_marker + maxLine, n_markers)
            # Strided view (no copy): the sgemm/sgemv/einsum below all accept
            # it and give the same result as on a contiguous copy.
            G_batch_f32 = chrom_geno_eigen_f32[:, start_marker:end_marker]
            batch_data = (
                G_batch_f32,
                weights_f32,
                XTW_f32,
                wy_f32,
                UXWUX_f64,
                UXWy_f64,
                vg_hat,
                start_marker,
            )
            _, batch_effects, batch_se, batch_pvals = process_batch_parallel(batch_data)
            effects[start_marker:end_marker] = batch_effects
            std_errors[start_marker:end_marker] = batch_se
            p_values[start_marker:end_marker] = batch_pvals

    return effects, std_errors, p_values


if HAS_NUMBA:

    @numba.jit(nopython=True, parallel=True, cache=True)
    def _wald_effects_trait_block_jit(
        UXWUs, UsWUs, UsWy, iUXWUX, iUXWUX_UXWy, vg, trait_ok, n, q0
    ):
        """Per-(trait, marker) Wald effect/SE from pre-reduced cross-products.

        Shapes: UXWUs (B, q0, mb) f32, UsWUs/UsWy (B, mb) f32, iUXWUX (B, q0, q0)
        f64, iUXWUX_UXWy (B, q0) f64, vg (B) f64, trait_ok (B) bool.
        Same arithmetic as process_batch_effects_jit, evaluated in float64.
        """
        B = UsWUs.shape[0]
        mb = UsWUs.shape[1]
        effects = np.zeros((B, mb))
        std_errors = np.zeros((B, mb))
        t_stats = np.zeros((B, mb))
        dfs = np.full((B, mb), float(n - q0 - 1))
        total = B * mb
        for idx in numba.prange(total):
            t = idx // mb
            j = idx - t * mb
            if not trait_ok[t]:
                effects[t, j] = np.nan
                std_errors[t, j] = np.nan
                t_stats[t, j] = np.nan
                dfs[t, j] = 1.0
                continue
            # u' iA u and u' (iA b), u = UXWUs[t, :, j]
            quad = 0.0
            lin = 0.0
            for a in range(q0):
                ua = float(UXWUs[t, a, j])
                acc = 0.0
                for b in range(q0):
                    acc += iUXWUX[t, a, b] * float(UXWUs[t, b, j])
                quad += ua * acc
                lin += ua * iUXWUX_UXWy[t, a]
            B22 = float(UsWUs[t, j]) - quad
            if B22 <= 1e-12:
                effects[t, j] = 0.0
                std_errors[t, j] = np.nan
                t_stats[t, j] = 0.0
                dfs[t, j] = 1.0
                continue
            invB22 = 1.0 / B22
            eff = invB22 * (float(UsWy[t, j]) - lin)
            se = np.sqrt(invB22 * vg[t])
            effects[t, j] = eff
            std_errors[t, j] = se
            if se > 0:
                t_stats[t, j] = eff / se
            else:
                t_stats[t, j] = 0.0
        return effects, std_errors, t_stats, dfs

else:

    def _wald_effects_trait_block_jit(
        UXWUs, UsWUs, UsWy, iUXWUX, iUXWUX_UXWy, vg, trait_ok, n, q0
    ):
        """NumPy fallback for the trait-block Wald kernel."""
        B, mb = UsWUs.shape
        U = UXWUs.astype(np.float64)
        quad = np.einsum("taj,tab,tbj->tj", U, iUXWUX, U)
        lin = np.einsum("taj,ta->tj", U, iUXWUX_UXWy)
        B22 = UsWUs.astype(np.float64) - quad
        good = (B22 > 1e-12) & trait_ok[:, None]
        invB22 = np.where(good, 1.0 / np.where(good, B22, 1.0), 0.0)
        effects = np.where(good, invB22 * (UsWy.astype(np.float64) - lin), 0.0)
        std_errors = np.where(good, np.sqrt(invB22 * vg[:, None]), np.nan)
        t_stats = np.where(good & (std_errors > 0), effects / np.where(good, std_errors, 1.0), 0.0)
        dfs = np.where(good, float(n - q0 - 1), 1.0)
        bad_trait = ~trait_ok[:, None] & np.ones((1, mb), dtype=bool)
        effects[bad_trait] = np.nan
        t_stats[bad_trait] = np.nan
        return effects, std_errors, t_stats, dfs


def _run_wald_trait_block(
    Y_transformed: np.ndarray,
    X_transformed: np.ndarray,
    eigenvals: np.ndarray,
    chrom_geno_eigen_f32: np.ndarray,
    *,
    vc_method: str,
    maxLine: int,
    cpu: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wald tests for a block of traits against one chromosome, batched as GEMMs.

    Each trait keeps its own REML variance ratio (so its own weight vector), but
    the three marker cross-products every trait needs reduce to three matrix
    products shared across the block, so the chromosome genotype block is
    streamed once per trait block instead of three times per trait:

      X'W_t G   for all t  ->  [stacked (X * w_t)'] @ G
      diag(G'W_t G)        ->  W' @ (G * G)
      G'W_t y_t            ->  (W * Y)' @ G

    Returns (effects, se, pvalues), each (n_traits_in_block, n_markers).
    """
    if vc_method.upper() != "BRENT":
        raise ValueError(f"Unknown variance component method: {vc_method}")

    n, B = Y_transformed.shape
    q0 = X_transformed.shape[1]
    n_markers = chrom_geno_eigen_f32.shape[1]
    if n_markers == 0:
        return (
            np.zeros((B, 0), dtype=np.float64),
            np.zeros((B, 0), dtype=np.float64),
            np.ones((B, 0), dtype=np.float64),
        )

    eig_safe = np.maximum(eigenvals, 1e-6)
    with numba_thread_limit(scaled_thread_count(cpu, B, 1)):
        delta, vg, _ = estimate_variance_components_brent_batch(
            Y_transformed, X_transformed, eigenvals
        )

    X_f32 = X_transformed.astype(np.float32, copy=False)
    Y_f32 = Y_transformed.astype(np.float32, copy=False)
    # W (n, B): per-trait eigenspace weights 1 / (lambda + delta_t)
    W_f32 = (1.0 / (eig_safe[:, None] + delta[None, :])).astype(np.float32)
    WY_f32 = W_f32 * Y_f32
    # Stacked weighted covariates: row block t is (X * w_t)' with shape (q0, n)
    XTW3_f32 = np.ascontiguousarray(X_f32.T[None, :, :] * W_f32.T[:, None, :])
    XTW_stack_f32 = XTW3_f32.reshape(B * q0, n)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            UXWUX_f32 = (XTW_stack_f32 @ X_f32).reshape(B, q0, q0)
            UXWy_f32 = np.einsum("tqn,nt->tq", XTW3_f32, Y_f32)
    if (not np.all(np.isfinite(UXWUX_f32))) or (not np.all(np.isfinite(UXWy_f32))):
        X64 = X_f32.astype(np.float64)
        XTW3_64 = XTW3_f32.astype(np.float64)
        UXWUX_f32 = (XTW3_64.reshape(B * q0, n) @ X64).reshape(B, q0, q0).astype(np.float32)
        UXWy_f32 = np.einsum("tqn,nt->tq", XTW3_64, Y_f32.astype(np.float64)).astype(np.float32)
    UXWUX = UXWUX_f32.astype(np.float64)
    UXWy = UXWy_f32.astype(np.float64)

    iUXWUX = np.zeros((B, q0, q0), dtype=np.float64)
    trait_ok = np.ones(B, dtype=np.bool_)
    for t in range(B):
        try:
            iUXWUX[t] = np.linalg.inv(UXWUX[t])
        except np.linalg.LinAlgError:
            trait_ok[t] = False
    iUXWUX_UXWy = np.einsum("tab,tb->ta", iUXWUX, UXWy)

    effects = np.zeros((B, n_markers), dtype=np.float64)
    std_errors = np.zeros((B, n_markers), dtype=np.float64)
    p_values = np.ones((B, n_markers), dtype=np.float64)
    n_batches = (n_markers + maxLine - 1) // maxLine

    with numba_thread_limit(scaled_thread_count(cpu, B * min(maxLine, n_markers), _WALD_MARKERS_PER_THREAD)):
        for batch_idx in range(n_batches):
            start = batch_idx * maxLine
            end = min(start + maxLine, n_markers)
            G_batch = chrom_geno_eigen_f32[:, start:end]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                UXWUs = (XTW_stack_f32 @ G_batch).reshape(B, q0, end - start)
                UsWUs = W_f32.T @ (G_batch * G_batch)
                UsWy = WY_f32.T @ G_batch
            eff, se, t_stats, dfs = _wald_effects_trait_block_jit(
                UXWUs, UsWUs, UsWy, iUXWUX, iUXWUX_UXWy, vg, trait_ok, n, q0
            )
            effects[:, start:end] = eff
            std_errors[:, start:end] = se
            p_values[:, start:end] = compute_fast_pvalues(t_stats, dfs, cpu=cpu)

    return effects, std_errors, p_values



def _refine_block_on_chromosome(
    *,
    Y_transformed: np.ndarray,
    X_transformed: np.ndarray,
    eigenvals: np.ndarray,
    eigenvecs_64: np.ndarray,
    geno: Union[GenotypeMatrix, np.ndarray],
    marker_indices: np.ndarray,
    effects: np.ndarray,
    std_errors: np.ndarray,
    p_values: np.ndarray,
    screen_threshold: float,
    lrt_solver_norm: str,
    lrt_batch_size: int,
    cpu: Optional[int],
) -> int:
    """Exact LRT refinement of the Wald hits of a trait block on one chromosome, in place.

    ``effects``/``std_errors``/``p_values`` are (traits, chromosome markers) and
    ``marker_indices`` are those columns' global positions. Cells with
    p < screen_threshold are refit with the exact LRT (same per-marker fits as
    the previous per-trait refinement; the ML null model per trait comes from
    the batched Brent kernel and candidate genotypes are fetched once per block
    rather than once per trait). Returns the number of refined cells.
    """
    cand_mask = p_values < screen_threshold
    n_candidates = int(np.count_nonzero(cand_mask))
    if n_candidates == 0:
        return 0

    eigenvals_lrt = np.maximum(
        np.nan_to_num(np.asarray(eigenvals, dtype=np.float64), nan=1e-6, posinf=1e6, neginf=1e-6),
        1e-6,
    )
    Y_lrt = np.nan_to_num(np.asarray(Y_transformed, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_lrt = np.nan_to_num(np.asarray(X_transformed, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    B = Y_lrt.shape[1]

    effective_cpu = scaled_thread_count(cpu, n_candidates, _LRT_MARKERS_PER_THREAD)
    with numba_thread_limit(effective_cpu):
        with numba_thread_limit(scaled_thread_count(cpu, B, 1)):
            h2_null, null_nll = estimate_null_ml_batch(Y_lrt, X_lrt, eigenvals_lrt)

        # Union of candidate columns across the block, fetched and transformed
        # once per chunk; each trait then takes the columns it screened.
        union_local = np.nonzero(cand_mask.any(axis=0))[0]
        chunk_n = max(1, int(lrt_batch_size))
        for c0 in range(0, union_local.size, chunk_n):
            local_cols = union_local[c0 : c0 + chunk_n]
            global_cols = marker_indices[local_cols]
            if isinstance(geno, GenotypeMatrix):
                g_raw = geno.get_columns_imputed(global_cols, dtype=np.float64)
            else:
                g_raw = impute_numpy_batch_major_allele(geno[:, global_cols], fill_value=None, dtype=np.float64)
            g_raw = np.nan_to_num(np.asarray(g_raw, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                g_eig = eigenvecs_64.T @ g_raw
            g_eig = np.nan_to_num(np.asarray(g_eig, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

            sub_mask = cand_mask[:, local_cols]
            for t in np.nonzero(sub_mask.any(axis=1))[0]:
                sel = np.nonzero(sub_mask[t])[0]
                g_t = np.ascontiguousarray(g_eig[:, sel])
                batch_p, batch_beta, batch_se = fit_markers_lrt_batch_prebuilt(
                    Y_lrt[:, t],
                    X_lrt,
                    g_t,
                    eigenvals_lrt,
                    float(null_nll[t]),
                    null_h2=float(h2_null[t]),
                    solver_norm=lrt_solver_norm,
                    assume_sanitized=True,
                )
                valid = np.isfinite(batch_p) & np.isfinite(batch_beta) & np.isfinite(batch_se)
                if not np.any(valid):
                    continue
                cols = local_cols[sel[valid]]
                p_values[t, cols] = batch_p[valid]
                effects[t, cols] = batch_beta[valid]
                std_errors[t, cols] = batch_se[valid]
    return n_candidates


def _process_chromosome(
    chrom: str,
    indices: np.ndarray,
    geno: Union[GenotypeMatrix, np.ndarray],
    phe: np.ndarray,
    loco_kinship: LocoKinship,
    CV: Optional[np.ndarray],
    vc_method: str,
    maxLine: int,
    cpu: int = 1,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process a single chromosome for LOCO MLM.

    Chromosomes are run sequentially; ``cpu`` is the marker-level parallelism
    budget handed to ``PANICLE_MLM`` for this chromosome's Wald scan.

    Returns:
        Tuple of (chrom, indices, effects, std_errors, pvalues)
    """
    geno_subset = _subset_genotypes(geno, indices)
    eigenK = loco_kinship.get_eigen(chrom)
    K_loco = loco_kinship.get_loco(chrom)

    # _subset_genotypes returns an imputed float32 block; for pre-imputed
    # GenotypeMatrix sources there is nothing to scan for, so tell PANICLE_MLM
    # to skip its two full-block -9/NaN passes.
    no_missing = isinstance(geno, GenotypeMatrix) and geno.is_imputed
    res = PANICLE_MLM(
        phe=phe,
        geno=geno_subset,
        K=K_loco,
        eigenK=eigenK,
        CV=CV,
        vc_method=vc_method,
        maxLine=maxLine,
        cpu=cpu,  # marker-level parallelism (chromosomes run sequentially)
        verbose=False,
        assume_no_missing=no_missing,
    )

    return chrom, indices, res.effects, res.se, res.pvalues


def PANICLE_MLM_LOCO(
    phe: np.ndarray,
    geno: Union[GenotypeMatrix, np.ndarray],
    map_data,
    loco_kinship: Optional[LocoKinship] = None,
    CV: Optional[np.ndarray] = None,
    vc_method: str = "BRENT",
    maxLine: int = 1000,
    cpu: int = 1,
    lrt_refinement: bool = True,
    screen_threshold: float = 5e-4,
    lrt_solver: str = "GEMMA",
    lrt_batch_size: int = 2048,
    verbose: bool = True,
) -> AssociationResults:
    """Run MLM with LOCO kinship matrices grouped by chromosome."""
    if not isinstance(phe, np.ndarray) or phe.ndim != 2 or phe.shape[1] != 2:
        raise ValueError("Phenotype must be numpy array with 2 columns [ID, trait_value]")
    trait_values_full = phe[:, 1].astype(np.float64)
    finite_trait = np.isfinite(trait_values_full)
    if not np.all(finite_trait):
        raise missing_values_error(
            "Phenotype",
            ~finite_trait,
            sample_ids=phe[:, 0],
            action="filter individuals before PANICLE_MLM_LOCO",
        )
    if CV is not None:
        if CV.shape[0] != phe.shape[0]:
            raise ValueError("Covariate matrix must have same number of rows as phenotypes")
        try:
            CV = CV.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            raise ValueError("Covariate matrix must contain numeric values")
        finite_cv = np.isfinite(CV).all(axis=1)
        if not np.all(finite_cv):
            raise missing_values_error(
                "Covariate matrix",
                ~finite_cv,
                sample_ids=phe[:, 0],
                action="filter individuals before PANICLE_MLM_LOCO",
            )

    geno = ensure_eager_genotype(geno)

    if isinstance(geno, GenotypeMatrix):
        n_individuals = geno.n_individuals
        n_markers = geno.n_markers
    elif isinstance(geno, np.ndarray):
        n_individuals = geno.shape[0]
        n_markers = geno.shape[1]
    else:
        raise ValueError("Genotype must be GenotypeMatrix or numpy array")

    if n_individuals != phe.shape[0]:
        raise ValueError("Number of phenotype observations must match number of genotype individuals")

    chrom_values, chrom_groups, _ = _resolve_chromosome_groups(map_data, n_markers)

    lrt_solver_norm = str(lrt_solver).strip().upper()
    if lrt_solver_norm not in {"GEMMA", "BRENT", "AUTO"}:
        raise ValueError("lrt_solver must be one of: 'GEMMA', 'BRENT', 'AUTO'")
    if lrt_batch_size < 1:
        raise ValueError("lrt_batch_size must be >= 1")

    if loco_kinship is None:
        loco_kinship = PANICLE_K_VanRaden_LOCO(geno, map_data, maxLine=maxLine, verbose=verbose)
    elif loco_kinship.n_individuals != n_individuals:
        raise ValueError(
            "LOCO kinship dimensions must match the filtered phenotype/genotype sample set; "
            "recompute LOCO kinship after dropping missing phenotype rows."
        )

    # Run through the chromosome-major kernel (one U'G transform per
    # chromosome, then the Wald scan on the transformed block). This is the
    # same code path the grouped multi-trait scan uses, so a trait analysed
    # alone and the same trait analysed in a group give identical numbers, and
    # it is faster than the previous per-batch transform in PANICLE_MLM.
    results = PANICLE_MLM_LOCO_MULTI(
        phe=trait_values_full[:, np.newaxis],
        geno=geno,
        map_data=map_data,
        trait_names=["trait"],
        loco_kinship=loco_kinship,
        CV=CV,
        vc_method=vc_method,
        maxLine=maxLine,
        cpu=cpu,
        lrt_refinement=lrt_refinement,
        screen_threshold=screen_threshold,
        lrt_solver=lrt_solver,
        lrt_batch_size=lrt_batch_size,
        verbose=verbose,
        keep="all",
    )
    return results["trait"]


def PANICLE_MLM_LOCO_MULTI(
    phe: np.ndarray,
    geno: Union[GenotypeMatrix, np.ndarray],
    map_data,
    trait_names: Optional[List[str]] = None,
    loco_kinship: Optional[LocoKinship] = None,
    CV: Optional[np.ndarray] = None,
    vc_method: str = "BRENT",
    maxLine: int = 1000,
    cpu: int = 1,
    lrt_refinement: bool = True,
    screen_threshold: float = 5e-4,
    lrt_solver: str = "GEMMA",
    lrt_batch_size: int = 2048,
    verbose: bool = True,
    trait_block_size: Optional[int] = None,
    keep: str = "hits",
    output_dir: Optional[Union[str, Path]] = None,
    p_threshold: Optional[float] = None,
    dense_dtype=np.float32,
) -> Union[Dict[str, AssociationResults], MultiTraitResultStore]:
    """Run LOCO MLM for many traits sharing the same genotype sample set.

    This executes chromosome-major to reuse expensive genotype eigenspace
    transforms across traits:
      - For each chromosome, compute U'G once.
      - For each trait, estimate variance components (per-trait REML ratio).
      - Run the Wald tests for a block of traits at once: the marker
        cross-products for every trait in the block are three GEMMs against the
        transformed chromosome block (see _run_wald_trait_block).
      - Optionally apply trait-specific exact LRT refinement.

    trait_block_size controls how many traits share one pass over the genotype
    block (default: up to 1024, reduced so a block's result arrays stay under
    ~1 GB).

    What is kept (results are produced block by block and never held for all
    traits at once unless asked):
      - keep="hits" (default): only cells with p <= p_threshold (default
        0.05 / n_markers, i.e. per-trait Bonferroni) plus each trait's top
        marker. Returns a MultiTraitResultStore (``.hits``, ``.trait_hits``,
        ``.trait_summary``); with output_dir the hit table streams to
        ``hits.csv`` as it is produced.
      - keep="all", output_dir=None: every cell in memory, returned as the
        classic {trait: AssociationResults} dict (3 x traits x markers float64;
        ~240 GB for 20k genes x 500k markers, so only for small trait sets).
      - keep="all" with output_dir: every cell streamed to ``effects.npy`` /
        ``se.npy`` (dense_dtype, default float32) and ``pvalues.npy`` (float64)
        memmaps of shape (traits, markers), e.g. for SuSiE fine mapping, plus
        the hit table. Returns a MultiTraitResultStore that maps trait name ->
        AssociationResults backed by the memmaps; reopen later with
        MultiTraitResultStore.open(output_dir).
    """
    if not isinstance(phe, np.ndarray) or phe.ndim != 2:
        raise ValueError("Phenotype matrix must be a 2D numpy array (n_individuals × n_traits)")
    if phe.shape[1] < 1:
        raise ValueError("Phenotype matrix must contain at least one trait column")

    trait_matrix = phe.astype(np.float64, copy=False)
    finite_trait_rows = np.isfinite(trait_matrix).all(axis=1)
    if not np.all(finite_trait_rows):
        raise missing_values_error(
            "Phenotype",
            ~finite_trait_rows,
            action="filter individuals before PANICLE_MLM_LOCO_MULTI",
        )

    n_individuals = trait_matrix.shape[0]
    n_traits = trait_matrix.shape[1]
    if trait_names is None:
        resolved_trait_names = [f"Trait{i + 1}" for i in range(n_traits)]
    else:
        resolved_trait_names = [str(name) for name in trait_names]
        if len(resolved_trait_names) != n_traits:
            raise ValueError("trait_names length must match number of phenotype columns")

    if CV is not None:
        if CV.shape[0] != n_individuals:
            raise ValueError("Covariate matrix must have same number of rows as phenotypes")
        try:
            CV = CV.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            raise ValueError("Covariate matrix must contain numeric values")
        finite_cv = np.isfinite(CV).all(axis=1)
        if not np.all(finite_cv):
            raise missing_values_error(
                "Covariate matrix",
                ~finite_cv,
                action="filter individuals before PANICLE_MLM_LOCO_MULTI",
            )

    geno = ensure_eager_genotype(geno)

    if isinstance(geno, GenotypeMatrix):
        if geno.n_individuals != n_individuals:
            raise ValueError("Number of phenotype observations must match number of genotype individuals")
        n_markers = geno.n_markers
    elif isinstance(geno, np.ndarray):
        if geno.shape[0] != n_individuals:
            raise ValueError("Number of phenotype observations must match number of genotype individuals")
        n_markers = geno.shape[1]
    else:
        raise ValueError("Genotype must be GenotypeMatrix or numpy array")

    chrom_values, chrom_groups, _ = _resolve_chromosome_groups(map_data, n_markers)
    chrom_items = [(chrom, indices) for chrom, indices in chrom_groups.items() if indices.size > 0]
    n_chroms = len(chrom_items)

    lrt_solver_norm = str(lrt_solver).strip().upper()
    if lrt_solver_norm not in {"GEMMA", "BRENT", "AUTO"}:
        raise ValueError("lrt_solver must be one of: 'GEMMA', 'BRENT', 'AUTO'")
    if lrt_batch_size < 1:
        raise ValueError("lrt_batch_size must be >= 1")

    if loco_kinship is None:
        loco_kinship = PANICLE_K_VanRaden_LOCO(geno, map_data, maxLine=maxLine, verbose=verbose)
    elif loco_kinship.n_individuals != n_individuals:
        raise ValueError(
            "LOCO kinship dimensions must match the filtered phenotype/genotype sample set; "
            "recompute LOCO kinship after dropping missing phenotype rows."
        )

    # Handle cpu=0 to mean use all available cores (affinity-aware)
    if cpu == 0:
        cpu = available_cpu_count()

    keep_norm = str(keep).lower()
    if keep_norm not in {"hits", "all"}:
        raise ValueError("keep must be 'hits' (sub-threshold cells only) or 'all'")
    if p_threshold is None:
        p_threshold = 0.05 / max(1, n_markers)
    p_threshold = float(p_threshold)
    legacy_dict = keep_norm == "all" and output_dir is None

    X = _build_design_matrix(n_individuals, CV)
    if legacy_dict:
        effects_by_trait = {name: np.zeros(n_markers, dtype=np.float64) for name in resolved_trait_names}
        std_errors_by_trait = {name: np.zeros(n_markers, dtype=np.float64) for name in resolved_trait_names}
        p_values_by_trait = {name: np.ones(n_markers, dtype=np.float64) for name in resolved_trait_names}
        store = None
    else:
        store = MultiTraitResultStore(
            resolved_trait_names,
            n_markers,
            keep=keep_norm,
            p_threshold=p_threshold,
            output_dir=output_dir,
            markers=marker_table(map_data, n_markers),
            dense_dtype=dense_dtype,
        )

    if verbose:
        print("=" * 60)
        print("LOCO MLM (MULTI-TRAIT)")
        print("=" * 60)
        print(f"Traits: {n_traits}")
        print(f"Chromosomes: {n_chroms}")
        if store is not None:
            print(f"Results: keep={keep_norm}, p_threshold={p_threshold:.3g}, "
                  f"output={'memory' if output_dir is None else output_dir}")

    # One float32 buffer, sized for the largest chromosome and reused for every
    # U'G product. A fresh multi-GB allocation per chromosome is returned to
    # the OS on free and re-page-faulted on the next chromosome (~1 s per GB);
    # writing into a buffer that is already touched avoids that and gives the
    # same bits (matmul(out=) vs a fresh result).
    max_chrom_markers = max((indices.size for _, indices in chrom_items), default=0)
    transform_buffer = np.empty(n_individuals * max_chrom_markers, dtype=np.float32)

    if trait_block_size is not None and int(trait_block_size) < 1:
        raise ValueError("trait_block_size must be >= 1")

    for chrom, indices in chrom_items:
        if trait_block_size is None:
            block = min(n_traits, _DEFAULT_TRAIT_BLOCK, max(1, _BLOCK_RESULT_BYTES // (24 * max(1, indices.size))))
        else:
            block = min(n_traits, int(trait_block_size))
        if verbose:
            print(
                f"Processing chromosome {chrom} ({indices.size} markers, {n_traits} traits, "
                f"blocks of {block})"
            )

        geno_subset = _subset_genotypes(geno, indices)
        eigen = loco_kinship.get_eigen(chrom)
        eigenvals = np.asarray(eigen["eigenvals"], dtype=np.float64)
        eigenvecs_32 = (
            np.asarray(eigen["eigenvecs"], dtype=np.float32)
            if np.asarray(eigen["eigenvecs"]).dtype != np.float32
            else np.asarray(eigen["eigenvecs"])
        )
        eigenvecs_64 = (
            eigenvecs_32.astype(np.float64)
            if eigenvecs_32.dtype != np.float64
            else eigenvecs_32
        )

        # Chromosome-major optimization: transform chromosome genotypes once,
        # into a C-contiguous prefix of the reused buffer.
        chrom_geno_eigen_f32 = transform_buffer[: n_individuals * indices.size].reshape(
            n_individuals, indices.size
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                np.matmul(
                    eigenvecs_32.T,
                    geno_subset.astype(np.float32, copy=False),
                    out=chrom_geno_eigen_f32,
                )
        # The untransformed float32 block is not needed after this point; drop
        # it so only one chromosome-sized float32 block is live per chromosome.
        del geno_subset

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                X_transformed = eigenvecs_64.T @ X
                Y_transformed = eigenvecs_64.T @ trait_matrix

        chrom_hits = 0
        chrom_refined = 0
        for block_start in range(0, n_traits, block):
            block_end = min(block_start + block, n_traits)
            block_eff, block_se, block_p = _run_wald_trait_block(
                Y_transformed=Y_transformed[:, block_start:block_end],
                X_transformed=X_transformed,
                eigenvals=eigenvals,
                chrom_geno_eigen_f32=chrom_geno_eigen_f32,
                vc_method=vc_method,
                maxLine=maxLine,
                cpu=cpu,
            )
            if lrt_refinement:
                chrom_refined += _refine_block_on_chromosome(
                    Y_transformed=Y_transformed[:, block_start:block_end],
                    X_transformed=X_transformed,
                    eigenvals=eigenvals,
                    eigenvecs_64=eigenvecs_64,
                    geno=geno,
                    marker_indices=indices,
                    effects=block_eff,
                    std_errors=block_se,
                    p_values=block_p,
                    screen_threshold=screen_threshold,
                    lrt_solver_norm=lrt_solver_norm,
                    lrt_batch_size=lrt_batch_size,
                    cpu=cpu,
                )
            if store is None:
                for offset, trait_idx in enumerate(range(block_start, block_end)):
                    trait_name = resolved_trait_names[trait_idx]
                    effects_by_trait[trait_name][indices] = block_eff[offset]
                    std_errors_by_trait[trait_name][indices] = block_se[offset]
                    p_values_by_trait[trait_name][indices] = block_p[offset]
            else:
                chrom_hits += store.write_block(block_start, indices, block_eff, block_se, block_p)
            del block_eff, block_se, block_p

        if verbose:
            msg = f"  chromosome {chrom} done"
            if lrt_refinement:
                msg += f"; LRT-refined {chrom_refined} cells (p < {screen_threshold})"
            if store is not None:
                msg += f"; {chrom_hits} hits (p <= {p_threshold:.3g})"
            print(msg)

        # Drop the view into the reused transform buffer before the next chromosome.
        del chrom_geno_eigen_f32
    del transform_buffer

    if store is not None:
        store.finalize()
        if verbose:
            print(f"Done: {store!r}")
        return store

    return {
        trait_name: AssociationResults(
            effects=effects_by_trait[trait_name],
            se=std_errors_by_trait[trait_name],
            pvalues=p_values_by_trait[trait_name],
        )
        for trait_name in resolved_trait_names
    }
