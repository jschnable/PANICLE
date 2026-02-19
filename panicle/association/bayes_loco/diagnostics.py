"""Diagnostics helpers for BAYESLOCO."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any
import numpy as np

from .state import FitCost, CaviState


def finalize_cost(cost: FitCost) -> None:
    cost.pass_equiv_total = (
        float(cost.pass_equiv_prior_tune)
        + float(cost.pass_equiv_main_fit)
        + float(cost.pass_equiv_loco_refine)
    )
    cost.timing_total_s = (
        float(cost.timing_prior_tune_s)
        + float(cost.timing_main_fit_s)
        + float(cost.timing_loco_test_s)
    )


def build_metadata(
    *,
    state: CaviState,
    cost: FitCost,
    h2_hat: float,
    pi_selected: float,
    slab_scale_selected: float,
    sigma_slab2_selected: float,
    prior_tuning_metric: str,
    prior_tuning_score: float,
    prior_stage1: int,
    prior_stage2: int,
    loco_mode: str,
    calibration_mode: str,
    lambda_gc_raw: float,
    lambda_gc_final: float,
    n_markers_fit: int,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    meta["method"] = "BAYESLOCO"
    meta["loco_mode"] = loco_mode
    meta["calibration_mode"] = calibration_mode
    meta["lambda_gc_raw"] = float(lambda_gc_raw)
    meta["lambda_gc_final"] = float(lambda_gc_final)
    meta["elbo_trace"] = [float(x) for x in state.elbo_trace]
    meta["converged"] = bool(state.converged)
    meta["sigma_e2_final"] = float(state.sigma_e2)
    meta["n_markers_fit"] = int(n_markers_fit)
    meta["h2_hat"] = float(h2_hat)
    meta["prior_pi_selected"] = float(pi_selected)
    meta["prior_slab_scale_selected"] = float(slab_scale_selected)
    meta["sigma_slab2_selected"] = float(sigma_slab2_selected)
    meta["prior_tuning_metric"] = prior_tuning_metric
    meta["prior_tuning_score"] = float(prior_tuning_score)
    meta["prior_tune_candidates_stage1"] = int(prior_stage1)
    meta["prior_tune_candidates_stage2"] = int(prior_stage2)
    meta["active_markers_trace"] = [int(x) for x in state.active_markers_trace]
    meta.update(asdict(cost))
    return meta


def compute_lambda_gc_from_stats(statistics: np.ndarray, eps: float = 1e-12) -> float:
    valid = np.asarray(statistics, dtype=np.float64)
    valid = valid[np.isfinite(valid) & (valid >= 0.0)]
    if valid.size == 0:
        return 1.0
    lam = float(np.median(valid) / 0.4549364)
    if not np.isfinite(lam) or lam < eps:
        return 1.0
    return lam
