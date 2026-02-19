"""Core BAYESLOCO fitting/testing engine."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2, norm

from ...utils.data_types import AssociationResults
from .config import BayesLocoConfig
from .data import BayesLocoData
from .diagnostics import build_metadata, compute_lambda_gc_from_stats, finalize_cost
from .state import CaviState, FitCost


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _safe_clip_scalar(x: float, bound: float = 1e6, default: float = 0.0) -> float:
    if not np.isfinite(x):
        return default
    return float(np.clip(x, -bound, bound))


def _effective_sigma_spike2(cfg: BayesLocoConfig, sigma_slab2: float) -> float:
    """Keep spike variance below slab variance to avoid prior inversion."""
    # If h2 is tiny, sigma_slab2 can become very small; cap spike accordingly.
    spike_cap = max(float(sigma_slab2) * 0.1, cfg.eps)
    return float(max(min(float(cfg.sigma_spike2), spike_cap), cfg.eps))


def _tau_for_epoch(cfg: BayesLocoConfig, epoch: int) -> float:
    if not cfg.kl_anneal:
        return 1.0
    return min(1.0, float(epoch + 1) / max(cfg.kl_anneal_epochs, 1))


@dataclass
class _FitRunResult:
    state: CaviState
    pass_equiv: float
    val_score: float
    pruned: bool


def _estimate_h2_he(data: BayesLocoData, cfg: BayesLocoConfig) -> float:
    """Approximate HE-style h2 estimate using sampled pairs and markers."""
    if data.n <= 2 or data.m_effective == 0:
        return float(np.clip(0.2, cfg.h2_min, cfg.h2_max))

    rng = np.random.default_rng(cfg.random_seed)
    m_sub = min(cfg.h2_marker_sample, data.m_effective)
    marker_sub = np.sort(rng.choice(data.fit_indices, size=m_sub, replace=False))
    Z_sub = data.get_standardized_block(marker_sub, dtype=np.float64)

    n_pairs = min(cfg.h2_pair_sample, data.n * (data.n - 1) // 2)
    if n_pairs <= 0:
        return float(np.clip(0.2, cfg.h2_min, cfg.h2_max))

    i = rng.integers(0, data.n, size=n_pairs, endpoint=False)
    j = rng.integers(0, data.n, size=n_pairs, endpoint=False)
    neq = i != j
    if not np.all(neq):
        i = i[neq]
        j = j[neq]
    if i.size == 0:
        return float(np.clip(0.2, cfg.h2_min, cfg.h2_max))

    k = np.mean(Z_sub[i, :] * Z_sub[j, :], axis=1)
    y_prod = data.r[i] * data.r[j]
    k_c = k - np.mean(k)
    y_c = y_prod - np.mean(y_prod)
    denom = float(np.dot(k_c, k_c))
    if denom <= cfg.eps:
        return float(np.clip(0.2, cfg.h2_min, cfg.h2_max))

    sigma_g2 = float(np.dot(k_c, y_c) / denom)
    h2_hat = sigma_g2 / max(data.var_r, cfg.eps)
    h2_hat = float(np.clip(h2_hat, cfg.h2_min, cfg.h2_max))
    if not np.isfinite(h2_hat):
        h2_hat = float(np.clip(0.2, cfg.h2_min, cfg.h2_max))
    return h2_hat


def _compute_sigma_slab2(
    *,
    h2_hat: float,
    var_r: float,
    pi: float,
    kappa: float,
    m_effective: int,
) -> float:
    return float(kappa * (h2_hat * var_r) / max(pi * float(m_effective), 1.0))


def _compute_validation_score(
    *,
    data: BayesLocoData,
    marker_indices: np.ndarray,
    m: np.ndarray,
    val_idx: np.ndarray,
    sigma_e2: float,
    metric: str,
    batch_size: int,
    eps: float,
) -> float:
    if val_idx.size == 0:
        return 0.0
    yhat = np.zeros(val_idx.size, dtype=np.float64)
    for start in range(0, marker_indices.size, batch_size):
        end = min(start + batch_size, marker_indices.size)
        idx = marker_indices[start:end]
        if idx.size == 0:
            continue
        Z = data.get_standardized_block(idx, row_index=val_idx, dtype=np.float64)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            yhat += np.nan_to_num(Z @ m[start:end], nan=0.0, posinf=0.0, neginf=0.0)
    resid = data.r[val_idx] - yhat
    if metric == "val_mse":
        return float(np.mean(resid * resid))
    sigma = max(float(sigma_e2), eps)
    # Mean Gaussian NLL up to additive constants shared across candidates.
    return float(0.5 * np.mean(np.log(2.0 * math.pi * sigma) + (resid * resid) / sigma))


def _compute_elbo(
    *,
    state: CaviState,
    r_resid: np.ndarray,
    d_std: np.ndarray,
    pi: float,
    sigma_slab2: float,
    sigma_spike2: float,
    tau: float,
    eps: float,
) -> float:
    n = r_resid.size
    sigma_e2 = max(float(state.sigma_e2), eps)
    rss = float(np.dot(r_resid, r_resid))
    v_term = float(np.dot(state.v, d_std))
    ll = -0.5 * n * np.log(2.0 * math.pi * sigma_e2) - 0.5 * (rss + v_term) / sigma_e2

    phi = np.clip(state.phi, eps, 1.0 - eps)
    pi0 = max(1.0 - pi, eps)
    kl_bern = np.sum(phi * np.log(phi / max(pi, eps)) + (1.0 - phi) * np.log((1.0 - phi) / pi0))

    s2_slab = np.maximum(state.s2_slab, eps)
    s2_spike = np.maximum(state.s2_spike, eps)
    kl_slab = 0.5 * np.sum(
        phi
        * (
            np.log(sigma_slab2 / s2_slab)
            + (s2_slab + state.mu_slab * state.mu_slab) / max(sigma_slab2, eps)
            - 1.0
        )
    )
    kl_spike = 0.5 * np.sum(
        (1.0 - phi)
        * (
            np.log(sigma_spike2 / s2_spike)
            + (s2_spike + state.mu_spike * state.mu_spike) / max(sigma_spike2, eps)
            - 1.0
        )
    )
    return float(ll - tau * (kl_bern + kl_slab + kl_spike))


def _initialize_state(
    n_markers: int,
    *,
    pi: float,
    sigma_slab2: float,
    sigma_spike2: float,
    sigma_e2_init: float,
    initial: Optional[CaviState] = None,
) -> CaviState:
    if initial is not None:
        return CaviState(
            m=np.array(initial.m, copy=True),
            v=np.array(initial.v, copy=True),
            phi=np.array(initial.phi, copy=True),
            mu_slab=np.array(initial.mu_slab, copy=True),
            s2_slab=np.array(initial.s2_slab, copy=True),
            mu_spike=np.array(initial.mu_spike, copy=True),
            s2_spike=np.array(initial.s2_spike, copy=True),
            sigma_e2=float(initial.sigma_e2),
            elbo_trace=[],
            active_markers_trace=[],
            converged=False,
            n_epochs=0,
        )
    return CaviState(
        m=np.zeros(n_markers, dtype=np.float64),
        v=np.zeros(n_markers, dtype=np.float64),
        phi=np.full(n_markers, float(pi), dtype=np.float64),
        mu_slab=np.zeros(n_markers, dtype=np.float64),
        s2_slab=np.full(n_markers, float(sigma_slab2), dtype=np.float64),
        mu_spike=np.zeros(n_markers, dtype=np.float64),
        s2_spike=np.full(n_markers, float(sigma_spike2), dtype=np.float64),
        sigma_e2=float(sigma_e2_init),
        elbo_trace=[],
        active_markers_trace=[],
        converged=False,
        n_epochs=0,
    )


def _fit_cavi(
    *,
    data: BayesLocoData,
    marker_indices: np.ndarray,
    pi: float,
    sigma_slab2: float,
    cfg: BayesLocoConfig,
    max_iter: int,
    patience: int,
    train_idx: Optional[np.ndarray] = None,
    val_idx: Optional[np.ndarray] = None,
    initial_state: Optional[CaviState] = None,
    enable_screening: bool = True,
    allow_marginal_init: bool = False,
    prune_after_epochs: Optional[int] = None,
    prune_best_score_ref: Optional[float] = None,
) -> _FitRunResult:
    if marker_indices.size == 0:
        raise ValueError("marker_indices is empty in CAVI fit")

    if train_idx is None:
        train_idx = np.arange(data.n, dtype=np.int64)
    if val_idx is None:
        val_idx = np.array([], dtype=np.int64)

    n_train = int(train_idx.size)
    r_train = np.asarray(data.r[train_idx], dtype=np.float64)
    sigma_e2_init = max(cfg.sigma_e2_min, float(np.var(r_train, ddof=1 if n_train > 1 else 0)))
    sigma_spike2_eff = _effective_sigma_spike2(cfg, sigma_slab2)

    # Precompute ||z_std||^2 in train space once.
    if train_idx.size == data.n:
        d_std = np.asarray(data.marker_d_std[marker_indices], dtype=np.float64)
    else:
        d_std = data.compute_d_std_subset(train_idx, marker_indices)
    d_std = np.maximum(d_std, cfg.eps)

    state = _initialize_state(
        marker_indices.size,
        pi=pi,
        sigma_slab2=sigma_slab2,
        sigma_spike2=sigma_spike2_eff,
        sigma_e2_init=sigma_e2_init,
        initial=initial_state,
    )

    # Optional marginal initialization for cold starts.
    if allow_marginal_init and initial_state is None and cfg.marginal_init:
        logit_pi = math.log(max(pi, cfg.eps) / max(1.0 - pi, cfg.eps))
        bsz = max(1, min(cfg.batch_markers_fit, marker_indices.size))
        for start in range(0, marker_indices.size, bsz):
            end = min(start + bsz, marker_indices.size)
            idx_abs = marker_indices[start:end]
            Z = data.get_standardized_block(idx_abs, row_index=train_idx, dtype=np.float64)
            with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                u = np.nan_to_num(Z.T @ r_train, nan=0.0, posinf=0.0, neginf=0.0)
            d = np.maximum(d_std[start:end], cfg.eps)
            beta = u / d
            t_abs = np.abs(u) / np.sqrt(np.maximum(d * state.sigma_e2, cfg.eps))
            logits = logit_pi + np.clip(t_abs - 1.5, -8.0, 8.0)
            phi = _sigmoid(logits)
            m = phi * beta
            state.phi[start:end] = phi
            state.mu_slab[start:end] = beta
            state.s2_slab[start:end] = np.maximum(state.sigma_e2 / d, cfg.eps)
            state.mu_spike[start:end] = 0.0
            state.s2_spike[start:end] = sigma_spike2_eff
            state.m[start:end] = m
            state.v[start:end] = np.maximum(phi * (state.s2_slab[start:end] + beta * beta) - m * m, 0.0)

    # Build residual from current state in one pass.
    r_resid = np.array(r_train, copy=True)
    bsz = max(1, min(cfg.batch_markers_fit, marker_indices.size))
    for start in range(0, marker_indices.size, bsz):
        end = min(start + bsz, marker_indices.size)
        idx_abs = marker_indices[start:end]
        Z = data.get_standardized_block(idx_abs, row_index=train_idx, dtype=np.float64)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            r_resid -= np.nan_to_num(Z @ state.m[start:end], nan=0.0, posinf=0.0, neginf=0.0)

    order_rng = np.random.default_rng(cfg.random_seed if cfg.deterministic else None)
    base_order = order_rng.permutation(marker_indices.size)

    best_elbo = -np.inf
    bad_epochs = 0
    pass_equiv = 0.0
    pruned = False

    for epoch in range(max_iter):
        screened = bool(enable_screening and cfg.active_set_screening and epoch >= cfg.screening_warmup_epochs)
        if not screened:
            active = base_order
        else:
            full_sweep = ((epoch - cfg.screening_warmup_epochs) % max(cfg.verification_interval, 1)) == 0
            if full_sweep:
                active = base_order
            else:
                mask = state.phi >= cfg.screening_threshold
                if cfg.screening_keep_top_k > 0:
                    keep_k = min(cfg.screening_keep_top_k, state.phi.size)
                    if keep_k > 0:
                        top = np.argpartition(state.phi, -keep_k)[-keep_k:]
                        mask[top] = True
                active = base_order[mask[base_order]]
                if active.size == 0:
                    active = base_order

        state.active_markers_trace.append(int(active.size))
        pass_equiv += float(active.size) / max(float(marker_indices.size), 1.0)

        for start_a in range(0, active.size, bsz):
            end_a = min(start_a + bsz, active.size)
            block_local = active[start_a:end_a]
            idx_abs = marker_indices[block_local]
            Z = data.get_standardized_block(idx_abs, row_index=train_idx, dtype=np.float64)

            for j in range(block_local.size):
                loc = int(block_local[j])
                z = Z[:, j]
                d = max(d_std[loc], cfg.eps)
                m_old = state.m[loc]
                if m_old != 0.0:
                    r_j = r_resid + z * m_old
                else:
                    r_j = r_resid
                u = _safe_clip_scalar(float(np.dot(z, r_j)))

                s2_slab = 1.0 / (d / max(state.sigma_e2, cfg.eps) + 1.0 / max(sigma_slab2, cfg.eps))
                mu_slab = _safe_clip_scalar(s2_slab * (u / max(state.sigma_e2, cfg.eps)))
                s2_spike = 1.0 / (d / max(state.sigma_e2, cfg.eps) + 1.0 / max(sigma_spike2_eff, cfg.eps))
                mu_spike = _safe_clip_scalar(s2_spike * (u / max(state.sigma_e2, cfg.eps)))

                logw1 = (
                    math.log(max(pi, cfg.eps))
                    + 0.5 * (math.log(max(s2_slab, cfg.eps) / max(sigma_slab2, cfg.eps)) + (mu_slab * mu_slab) / max(s2_slab, cfg.eps))
                )
                logw0 = (
                    math.log(max(1.0 - pi, cfg.eps))
                    + 0.5 * (math.log(max(s2_spike, cfg.eps) / max(sigma_spike2_eff, cfg.eps)) + (mu_spike * mu_spike) / max(s2_spike, cfg.eps))
                )
                phi = float(_sigmoid(logw1 - logw0))
                m_new = _safe_clip_scalar(phi * mu_slab + (1.0 - phi) * mu_spike)
                second = phi * (s2_slab + mu_slab * mu_slab) + (1.0 - phi) * (s2_spike + mu_spike * mu_spike)
                v_new = max(_safe_clip_scalar(second - m_new * m_new, bound=1e12, default=0.0), 0.0)

                if cfg.cavi_damping < 1.0:
                    m_new = cfg.cavi_damping * m_new + (1.0 - cfg.cavi_damping) * m_old

                delta = m_new - m_old
                if delta != 0.0:
                    r_resid -= z * delta

                state.m[loc] = m_new
                state.v[loc] = v_new
                state.phi[loc] = phi
                state.mu_slab[loc] = mu_slab
                state.s2_slab[loc] = max(s2_slab, cfg.eps)
                state.mu_spike[loc] = mu_spike
                state.s2_spike[loc] = max(s2_spike, cfg.eps)

        if cfg.estimate_sigma_e2:
            rss = float(np.dot(r_resid, r_resid))
            vterm = float(np.dot(state.v, d_std))
            state.sigma_e2 = max(cfg.sigma_e2_min, (rss + vterm) / max(float(n_train), 1.0))

        interval = cfg.elbo_eval_interval_screened if screened else cfg.elbo_eval_interval
        if (epoch % interval == 0) or (epoch == max_iter - 1):
            tau = _tau_for_epoch(cfg, epoch)
            elbo = _compute_elbo(
                state=state,
                r_resid=r_resid,
                d_std=d_std,
                pi=pi,
                sigma_slab2=sigma_slab2,
                sigma_spike2=sigma_spike2_eff,
                tau=tau,
                eps=cfg.eps,
            )
            state.elbo_trace.append(elbo)
            if elbo > best_elbo + cfg.tol_elbo:
                best_elbo = elbo
                bad_epochs = 0
            else:
                bad_epochs += 1

        if (
            prune_after_epochs is not None
            and val_idx.size > 0
            and (epoch + 1) == int(prune_after_epochs)
            and prune_best_score_ref is not None
            and np.isfinite(prune_best_score_ref)
        ):
            mid_score = _compute_validation_score(
                data=data,
                marker_indices=marker_indices,
                m=state.m,
                val_idx=val_idx,
                sigma_e2=state.sigma_e2,
                metric=cfg.prior_tune_metric,
                batch_size=cfg.batch_markers_fit,
                eps=cfg.eps,
            )
            if mid_score > prune_best_score_ref * (1.0 + cfg.prior_tune_prune_rel_gap):
                pruned = True
                state.n_epochs = epoch + 1
                break

        if bad_epochs >= patience:
            state.converged = True
            state.n_epochs = epoch + 1
            break
    else:
        state.n_epochs = max_iter

    val_score = _compute_validation_score(
        data=data,
        marker_indices=marker_indices,
        m=state.m,
        val_idx=val_idx,
        sigma_e2=state.sigma_e2,
        metric=cfg.prior_tune_metric,
        batch_size=cfg.batch_markers_fit,
        eps=cfg.eps,
    )
    return _FitRunResult(state=state, pass_equiv=pass_equiv, val_score=val_score, pruned=pruned)


def _state_subset(state: CaviState, local_indices: np.ndarray) -> CaviState:
    """Create a deep copied state subset over local marker indices."""
    return CaviState(
        m=np.array(state.m[local_indices], copy=True),
        v=np.array(state.v[local_indices], copy=True),
        phi=np.array(state.phi[local_indices], copy=True),
        mu_slab=np.array(state.mu_slab[local_indices], copy=True),
        s2_slab=np.array(state.s2_slab[local_indices], copy=True),
        mu_spike=np.array(state.mu_spike[local_indices], copy=True),
        s2_spike=np.array(state.s2_spike[local_indices], copy=True),
        sigma_e2=float(state.sigma_e2),
        elbo_trace=[],
        active_markers_trace=[],
        converged=False,
        n_epochs=0,
    )


def _prior_tune(
    *,
    data: BayesLocoData,
    cfg: BayesLocoConfig,
    h2_hat: float,
    cost: FitCost,
    verbose: bool,
) -> Tuple[float, float, float, float, Optional[CaviState], int, int]:
    split = data.split_train_val()
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    var_r_train = float(np.var(data.r[train_idx], ddof=1 if train_idx.size > 1 else 0))

    candidates: List[Tuple[float, float]] = []
    for pi in sorted(cfg.prior_tune_pi_grid):
        for kappa in sorted(cfg.prior_tune_slab_scale_grid):
            candidates.append((float(pi), float(kappa)))

    rng = np.random.default_rng(cfg.random_seed)
    stage1_candidates = candidates
    stage2_candidates: List[Tuple[float, float]]
    n_stage1 = len(candidates)
    n_stage2 = len(candidates)

    best_global_score = np.inf

    if cfg.prior_tune_two_stage:
        m1 = max(1, int(round(data.m_effective * cfg.prior_tune_stage1_marker_fraction)))
        marker_perm = rng.permutation(data.fit_indices)
        stage1_markers = np.sort(marker_perm[:m1])

        scored_stage1: List[Tuple[float, float, float]] = []
        prev_state = None
        for pi, kappa in stage1_candidates:
            sigma_slab2 = _compute_sigma_slab2(
                h2_hat=h2_hat,
                var_r=var_r_train,
                pi=pi,
                kappa=kappa,
                m_effective=data.m_effective,
            )
            run = _fit_cavi(
                data=data,
                marker_indices=stage1_markers,
                pi=pi,
                sigma_slab2=sigma_slab2,
                cfg=cfg,
                max_iter=cfg.prior_tune_stage1_max_iter,
                patience=cfg.prior_tune_stage1_patience,
                train_idx=train_idx,
                val_idx=val_idx,
                initial_state=prev_state if cfg.prior_tune_warm_start else None,
                enable_screening=False,
                allow_marginal_init=True,
                prune_after_epochs=cfg.prior_tune_prune_after_epochs,
                prune_best_score_ref=(best_global_score if np.isfinite(best_global_score) else None),
            )
            # Convert local pass-equivalent (relative to stage1 markers) to full-fit equivalent.
            cost.pass_equiv_prior_tune += run.pass_equiv * (float(stage1_markers.size) / max(float(data.m_effective), 1.0))
            scored_stage1.append((run.val_score, pi, kappa))
            if run.val_score < best_global_score:
                best_global_score = run.val_score
            prev_state = run.state if cfg.prior_tune_warm_start else None

        scored_stage1.sort(key=lambda x: (x[0], x[1], x[2]))
        keep_k = min(cfg.prior_tune_top_k, len(scored_stage1))
        stage2_candidates = [(x[1], x[2]) for x in scored_stage1[:keep_k]]
        n_stage2 = len(stage2_candidates)
    else:
        stage2_candidates = candidates

    best_pi = float(stage2_candidates[0][0])
    best_kappa = float(stage2_candidates[0][1])
    best_sigma_slab2 = _compute_sigma_slab2(
        h2_hat=h2_hat,
        var_r=var_r_train,
        pi=best_pi,
        kappa=best_kappa,
        m_effective=data.m_effective,
    )
    best_score = np.inf
    best_state: Optional[CaviState] = None

    prev_state = None
    max_iter = cfg.prior_tune_stage2_max_iter if cfg.prior_tune_two_stage else cfg.prior_tune_max_iter
    patience = cfg.prior_tune_stage2_patience if cfg.prior_tune_two_stage else cfg.prior_tune_patience

    for pi, kappa in stage2_candidates:
        sigma_slab2 = _compute_sigma_slab2(
            h2_hat=h2_hat,
            var_r=var_r_train,
            pi=pi,
            kappa=kappa,
            m_effective=data.m_effective,
        )
        run = _fit_cavi(
            data=data,
            marker_indices=data.fit_indices,
            pi=pi,
            sigma_slab2=sigma_slab2,
            cfg=cfg,
            max_iter=max_iter,
            patience=patience,
            train_idx=train_idx,
            val_idx=val_idx,
            initial_state=prev_state if cfg.prior_tune_warm_start else None,
            enable_screening=True,
            allow_marginal_init=True,
            prune_after_epochs=None,
            prune_best_score_ref=None,
        )
        cost.pass_equiv_prior_tune += run.pass_equiv
        if run.val_score < best_score or (
            np.isclose(run.val_score, best_score) and (pi < best_pi or (pi == best_pi and kappa < best_kappa))
        ):
            best_score = run.val_score
            best_pi = float(pi)
            best_kappa = float(kappa)
            best_sigma_slab2 = float(sigma_slab2)
            best_state = run.state
        prev_state = run.state if cfg.prior_tune_warm_start else None

    if verbose:
        print(
            f"  Prior tuning selected pi={best_pi:.4g}, slab_scale={best_kappa:.4g}, "
            f"sigma_slab2={best_sigma_slab2:.4g}, score={best_score:.5g}"
        )

    return best_pi, best_kappa, best_sigma_slab2, float(best_score), best_state, n_stage1, n_stage2


def _compute_loco_prediction_and_uncertainty(
    *,
    data: BayesLocoData,
    fit_indices: np.ndarray,
    m_fit: np.ndarray,
    v_fit: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, Dict[str, float]]:
    yhat_total = np.zeros(data.n, dtype=np.float64)
    yhat_chr: Dict[str, np.ndarray] = {chrom: np.zeros(data.n, dtype=np.float64) for chrom in data.chrom_order}
    v_total = 0.0
    v_chr: Dict[str, float] = {chrom: 0.0 for chrom in data.chrom_order}

    for start in range(0, fit_indices.size, batch_size):
        end = min(start + batch_size, fit_indices.size)
        idx_abs = fit_indices[start:end]
        if idx_abs.size == 0:
            continue
        Z = data.get_standardized_block(idx_abs, dtype=np.float64)
        m_block = m_fit[start:end]
        v_block = v_fit[start:end]
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            yhat_total += np.nan_to_num(Z @ m_block, nan=0.0, posinf=0.0, neginf=0.0)

        d_std = data.marker_d_std[idx_abs]
        v_total += float(np.dot(v_block, d_std))

        chrom_block = data.chrom_values[idx_abs]
        unique_chroms = np.unique(chrom_block)
        for chrom in unique_chroms:
            mask = chrom_block == chrom
            if not np.any(mask):
                continue
            with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                yhat_chr[str(chrom)] += np.nan_to_num(Z[:, mask] @ m_block[mask], nan=0.0, posinf=0.0, neginf=0.0)
            v_chr[str(chrom)] += float(np.dot(v_block[mask], d_std[mask]))

    return yhat_total, yhat_chr, float(v_total), v_chr


def _predict_from_state(
    *,
    data: BayesLocoData,
    marker_indices: np.ndarray,
    state: CaviState,
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    """Predict phenotype and posterior uncertainty contribution for selected markers."""
    yhat = np.zeros(data.n, dtype=np.float64)
    v_sum = 0.0
    bsz = max(1, min(batch_size, marker_indices.size if marker_indices.size > 0 else 1))
    for start in range(0, marker_indices.size, bsz):
        end = min(start + bsz, marker_indices.size)
        idx_abs = marker_indices[start:end]
        if idx_abs.size == 0:
            continue
        Z = data.get_standardized_block(idx_abs, dtype=np.float64)
        m_block = state.m[start:end]
        v_block = state.v[start:end]
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            yhat += np.nan_to_num(Z @ m_block, nan=0.0, posinf=0.0, neginf=0.0)
        d_std = data.marker_d_std[idx_abs]
        v_sum += float(np.dot(v_block, d_std))
    return yhat, float(v_sum)


def _estimate_sigma_test2_from_residual(
    *,
    residual: np.ndarray,
    fallback_sigma_e2: float,
    cfg: BayesLocoConfig,
) -> float:
    """Estimate test-time residual variance in LOCO space.

    We use the empirical variance of LOCO residuals so score/Wald tests are
    calibrated in the same space where the association statistic is computed.
    """
    ddof = 1 if residual.size > 1 else 0
    sigma_resid = float(np.var(residual, ddof=ddof))
    if not np.isfinite(sigma_resid) or sigma_resid <= cfg.eps:
        sigma_resid = float(fallback_sigma_e2)
    return max(float(sigma_resid), cfg.sigma_e2_min)


def _run_tests(
    *,
    data: BayesLocoData,
    cfg: BayesLocoConfig,
    sigma_e2: float,
    yhat_total: np.ndarray,
    yhat_chr: Dict[str, np.ndarray],
    v_total: float,
    v_chr: Dict[str, float],
    unrelated_subset: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    effects = np.zeros(data.m, dtype=np.float64)
    ses = np.full(data.m, np.nan, dtype=np.float64)
    pvals = np.ones(data.m, dtype=np.float64)
    chi2_stats = np.zeros(data.m, dtype=np.float64)
    unrelated_ratios: List[np.ndarray] = []

    if unrelated_subset is not None:
        unrelated_subset = np.asarray(unrelated_subset, dtype=np.int64)
        unrelated_subset = unrelated_subset[(unrelated_subset >= 0) & (unrelated_subset < data.n)]
        if unrelated_subset.size == 0:
            unrelated_subset = None

    for chrom in data.chrom_order:
        idx_chr = data.chrom_groups.get(chrom, np.array([], dtype=np.int64))
        if idx_chr.size == 0:
            continue

        yhat_not_c = yhat_total - yhat_chr.get(chrom, 0.0)
        r_c = data.r - yhat_not_c

        sigma_base2 = _estimate_sigma_test2_from_residual(
            residual=r_c,
            fallback_sigma_e2=sigma_e2,
            cfg=cfg,
        )
        if cfg.residual_var_correction == "diag":
            v_uncert = (v_total - v_chr.get(chrom, 0.0)) / max(float(data.n), 1.0)
            sigma_test2 = sigma_base2 * (1.0 + v_uncert / max(sigma_base2, cfg.eps))
        else:
            sigma_test2 = sigma_base2
        sigma_test2 = max(float(sigma_test2), cfg.sigma_e2_min)

        bsz = max(1, min(cfg.batch_markers_test, idx_chr.size))
        for start in range(0, idx_chr.size, bsz):
            end = min(start + bsz, idx_chr.size)
            idx = idx_chr[start:end]
            Z = data.get_unstandardized_block(idx, dtype=np.float64)
            with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                u = np.nan_to_num(Z.T @ r_c, nan=0.0, posinf=0.0, neginf=0.0)
            d = np.maximum(data.marker_d_unstd[idx], cfg.eps)
            beta = u / d
            se = np.sqrt(sigma_test2 / d)
            chi2_block = (beta / np.maximum(se, cfg.eps)) ** 2

            if cfg.test_method == "score":
                score_stat = (u * u) / (sigma_test2 * d)
                p = chi2.sf(score_stat, 1)
                chi2_block = score_stat
            else:
                if cfg.robust_se:
                    # HC1 robust SE for 1-df residualized OLS per marker.
                    e = r_c[:, np.newaxis] - Z * beta[np.newaxis, :]
                    meat = np.einsum("ij,ij->j", Z * e, Z * e)
                    hc0 = meat / np.maximum(d * d, cfg.eps)
                    n_obs = max(float(data.n), 2.0)
                    hc1 = hc0 * (n_obs / max(n_obs - 1.0, 1.0))
                    se = np.sqrt(np.maximum(hc1, cfg.eps))
                z = beta / np.maximum(se, cfg.eps)
                p = 2.0 * norm.sf(np.abs(z))
                chi2_block = z * z

            effects[idx] = beta
            ses[idx] = se
            pvals[idx] = np.clip(p, 0.0, 1.0)
            chi2_stats[idx] = np.maximum(chi2_block, 0.0)

            if unrelated_subset is not None and unrelated_subset.size > 1:
                Z_sub = Z[unrelated_subset, :]
                r_sub = r_c[unrelated_subset]
                d_sub = np.maximum(np.einsum("ij,ij->j", Z_sub, Z_sub), cfg.eps)
                with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                    u_sub = np.nan_to_num(Z_sub.T @ r_sub, nan=0.0, posinf=0.0, neginf=0.0)
                sigma_sub2 = max(float(np.var(r_sub, ddof=1)), cfg.sigma_e2_min)
                chi2_ref = (u_sub * u_sub) / (sigma_sub2 * d_sub)
                valid = np.isfinite(chi2_ref) & (chi2_ref > cfg.eps) & np.isfinite(chi2_block)
                if np.any(valid):
                    unrelated_ratios.append(np.asarray(chi2_block[valid] / chi2_ref[valid], dtype=np.float64))

    unrelated_scale = float("nan")
    if unrelated_ratios:
        ratios = np.concatenate(unrelated_ratios)
        ratios = ratios[np.isfinite(ratios) & (ratios > cfg.eps)]
        if ratios.size > 0:
            unrelated_scale = float(np.median(ratios))

    return effects, ses, pvals, chi2_stats, unrelated_scale


def _run_tests_refine(
    *,
    data: BayesLocoData,
    cfg: BayesLocoConfig,
    pi: float,
    sigma_slab2: float,
    base_state: CaviState,
    cost: FitCost,
    unrelated_subset: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run LOCO testing with per-chromosome refinement fits."""
    effects = np.zeros(data.m, dtype=np.float64)
    ses = np.full(data.m, np.nan, dtype=np.float64)
    pvals = np.ones(data.m, dtype=np.float64)
    chi2_stats = np.zeros(data.m, dtype=np.float64)
    unrelated_ratios: List[np.ndarray] = []

    fit_indices = data.fit_indices
    fit_chrom = data.chrom_values[fit_indices]

    if unrelated_subset is not None:
        unrelated_subset = np.asarray(unrelated_subset, dtype=np.int64)
        unrelated_subset = unrelated_subset[(unrelated_subset >= 0) & (unrelated_subset < data.n)]
        if unrelated_subset.size == 0:
            unrelated_subset = None

    for chrom in data.chrom_order:
        idx_chr = data.chrom_groups.get(chrom, np.array([], dtype=np.int64))
        if idx_chr.size == 0:
            continue

        local_not_c = np.where(fit_chrom != chrom)[0].astype(np.int64)
        if local_not_c.size == 0:
            continue
        idx_not_c = fit_indices[local_not_c]

        init_state = _state_subset(base_state, local_not_c)
        refine_run = _fit_cavi(
            data=data,
            marker_indices=idx_not_c,
            pi=pi,
            sigma_slab2=sigma_slab2,
            cfg=cfg,
            max_iter=cfg.loco_refine_iter,
            patience=cfg.refine_patience,
            train_idx=np.arange(data.n, dtype=np.int64),
            val_idx=np.array([], dtype=np.int64),
            initial_state=init_state,
            enable_screening=True,
            allow_marginal_init=False,
        )
        cost.pass_equiv_loco_refine += refine_run.pass_equiv * (float(idx_not_c.size) / max(float(data.m_effective), 1.0))
        yhat_not_c, v_not_c = _predict_from_state(
            data=data,
            marker_indices=idx_not_c,
            state=refine_run.state,
            batch_size=cfg.batch_markers_fit,
        )
        r_c = data.r - yhat_not_c

        sigma_base2 = _estimate_sigma_test2_from_residual(
            residual=r_c,
            fallback_sigma_e2=refine_run.state.sigma_e2,
            cfg=cfg,
        )
        if cfg.residual_var_correction == "diag":
            v_uncert = v_not_c / max(float(data.n), 1.0)
            sigma_test2 = sigma_base2 * (1.0 + v_uncert / max(sigma_base2, cfg.eps))
        else:
            sigma_test2 = sigma_base2
        sigma_test2 = max(float(sigma_test2), cfg.sigma_e2_min)

        bsz = max(1, min(cfg.batch_markers_test, idx_chr.size))
        for start in range(0, idx_chr.size, bsz):
            end = min(start + bsz, idx_chr.size)
            idx = idx_chr[start:end]
            Z = data.get_unstandardized_block(idx, dtype=np.float64)
            with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                u = np.nan_to_num(Z.T @ r_c, nan=0.0, posinf=0.0, neginf=0.0)
            d = np.maximum(data.marker_d_unstd[idx], cfg.eps)
            beta = u / d
            se = np.sqrt(sigma_test2 / d)
            chi2_block = (beta / np.maximum(se, cfg.eps)) ** 2

            if cfg.test_method == "score":
                score_stat = (u * u) / (sigma_test2 * d)
                p = chi2.sf(score_stat, 1)
                chi2_block = score_stat
            else:
                if cfg.robust_se:
                    e = r_c[:, np.newaxis] - Z * beta[np.newaxis, :]
                    meat = np.einsum("ij,ij->j", Z * e, Z * e)
                    hc0 = meat / np.maximum(d * d, cfg.eps)
                    n_obs = max(float(data.n), 2.0)
                    hc1 = hc0 * (n_obs / max(n_obs - 1.0, 1.0))
                    se = np.sqrt(np.maximum(hc1, cfg.eps))
                z = beta / np.maximum(se, cfg.eps)
                p = 2.0 * norm.sf(np.abs(z))
                chi2_block = z * z

            effects[idx] = beta
            ses[idx] = se
            pvals[idx] = np.clip(p, 0.0, 1.0)
            chi2_stats[idx] = np.maximum(chi2_block, 0.0)

            if unrelated_subset is not None and unrelated_subset.size > 1:
                Z_sub = Z[unrelated_subset, :]
                r_sub = r_c[unrelated_subset]
                d_sub = np.maximum(np.einsum("ij,ij->j", Z_sub, Z_sub), cfg.eps)
                with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                    u_sub = np.nan_to_num(Z_sub.T @ r_sub, nan=0.0, posinf=0.0, neginf=0.0)
                sigma_sub2 = max(float(np.var(r_sub, ddof=1)), cfg.sigma_e2_min)
                chi2_ref = (u_sub * u_sub) / (sigma_sub2 * d_sub)
                valid = np.isfinite(chi2_ref) & (chi2_ref > cfg.eps) & np.isfinite(chi2_block)
                if np.any(valid):
                    unrelated_ratios.append(np.asarray(chi2_block[valid] / chi2_ref[valid], dtype=np.float64))

    unrelated_scale = float("nan")
    if unrelated_ratios:
        ratios = np.concatenate(unrelated_ratios)
        ratios = ratios[np.isfinite(ratios) & (ratios > cfg.eps)]
        if ratios.size > 0:
            unrelated_scale = float(np.median(ratios))

    return effects, ses, pvals, chi2_stats, unrelated_scale


def _apply_stat_calibration(
    *,
    cfg: BayesLocoConfig,
    pvals: np.ndarray,
    chi2_stats: np.ndarray,
    unrelated_scale: float,
    verbose: bool,
) -> Tuple[np.ndarray, float, float, str]:
    mode = cfg.calibrate_stat_scale
    lambda_gc_raw = compute_lambda_gc_from_stats(chi2_stats, eps=cfg.eps)
    lambda_gc_final = lambda_gc_raw
    calibrated = np.array(pvals, copy=True)
    calibration_mode = mode

    if mode == "none":
        return calibrated, lambda_gc_raw, lambda_gc_final, calibration_mode

    if mode == "unrelated_subset":
        if np.isfinite(unrelated_scale) and unrelated_scale > cfg.eps:
            adj_stats = chi2_stats / unrelated_scale
            calibrated = chi2.sf(np.maximum(adj_stats, 0.0), 1)
            lambda_gc_final = compute_lambda_gc_from_stats(adj_stats, eps=cfg.eps)
            return np.clip(calibrated, 0.0, 1.0), lambda_gc_raw, lambda_gc_final, calibration_mode
        calibration_mode = "gc"
        if verbose:
            warnings.warn("unrelated_subset calibration failed to estimate stable scale; falling back to genomic control")

    if calibration_mode == "gc":
        lam = max(lambda_gc_raw, 1.0)
        adj_stats = chi2_stats / max(lam, cfg.eps)
        calibrated = chi2.sf(np.maximum(adj_stats, 0.0), 1)
        lambda_gc_final = compute_lambda_gc_from_stats(adj_stats, eps=cfg.eps)

    return np.clip(calibrated, 0.0, 1.0), lambda_gc_raw, lambda_gc_final, calibration_mode


def run_bayes_loco(
    *,
    phe: np.ndarray,
    geno,
    map_data,
    CV: Optional[np.ndarray],
    cpu: int,
    verbose: bool,
    cfg: BayesLocoConfig,
) -> AssociationResults:
    del cpu  # CPU count hooks reserved for future backend parallelism.

    t_total = time.perf_counter()
    cost = FitCost()

    data = BayesLocoData(phe=phe, geno=geno, map_data=map_data, CV=CV, cfg=cfg)
    unrelated_subset: Optional[np.ndarray] = None
    if cfg.calibrate_stat_scale == "unrelated_subset":
        if cfg.unrelated_subset_indices is None:
            raise ValueError(
                "calibrate_stat_scale='unrelated_subset' requires unrelated_subset_indices in bl_config"
            )
        unrelated_subset = np.asarray(cfg.unrelated_subset_indices, dtype=np.int64)
        unrelated_subset = unrelated_subset[(unrelated_subset >= 0) & (unrelated_subset < data.n)]
        unrelated_subset = np.unique(unrelated_subset)
        if unrelated_subset.size < cfg.unrelated_subset_min_n:
            raise ValueError(
                "unrelated_subset_indices is too small for unrelated_subset calibration: "
                f"{unrelated_subset.size} < {cfg.unrelated_subset_min_n}"
            )

    if verbose:
        print("=" * 60)
        print("BAYESLOCO")
        print("=" * 60)
        print(f"Samples: {data.n}")
        print(f"Markers: {data.m} (fit markers after QC: {data.m_effective})")

    h2_hat = _estimate_h2_he(data, cfg)
    if verbose:
        print(f"  Estimated h2: {h2_hat:.4f}")

    # Mandatory prior tuning.
    t0 = time.perf_counter()
    (
        pi_selected,
        slab_scale_selected,
        sigma_slab2_selected,
        prior_tuning_score,
        tune_state,
        prior_stage1,
        prior_stage2,
    ) = _prior_tune(data=data, cfg=cfg, h2_hat=h2_hat, cost=cost, verbose=verbose)
    cost.timing_prior_tune_s = time.perf_counter() - t0

    # Main full-data fit.
    t1 = time.perf_counter()
    fit = _fit_cavi(
        data=data,
        marker_indices=data.fit_indices,
        pi=pi_selected,
        sigma_slab2=sigma_slab2_selected,
        cfg=cfg,
        max_iter=cfg.max_iter,
        patience=cfg.patience,
        train_idx=np.arange(data.n, dtype=np.int64),
        val_idx=np.array([], dtype=np.int64),
        initial_state=tune_state if cfg.prior_tune_warm_start else None,
        enable_screening=True,
        allow_marginal_init=True,
    )
    cost.timing_main_fit_s = time.perf_counter() - t1
    cost.pass_equiv_main_fit += fit.pass_equiv

    # LOCO prediction and testing.
    t2 = time.perf_counter()
    if cfg.loco_mode == "refine":
        effects, ses, pvals, chi2_stats, unrelated_scale = _run_tests_refine(
            data=data,
            cfg=cfg,
            pi=pi_selected,
            sigma_slab2=sigma_slab2_selected,
            base_state=fit.state,
            cost=cost,
            unrelated_subset=unrelated_subset,
        )
        loco_mode_effective = "refine"
    else:
        yhat_total, yhat_chr, v_total, v_chr = _compute_loco_prediction_and_uncertainty(
            data=data,
            fit_indices=data.fit_indices,
            m_fit=fit.state.m,
            v_fit=fit.state.v,
            batch_size=cfg.batch_markers_fit,
        )
        effects, ses, pvals, chi2_stats, unrelated_scale = _run_tests(
            data=data,
            cfg=cfg,
            sigma_e2=fit.state.sigma_e2,
            yhat_total=yhat_total,
            yhat_chr=yhat_chr,
            v_total=v_total,
            v_chr=v_chr,
            unrelated_subset=unrelated_subset,
        )
        loco_mode_effective = "subtract_only"

    pvals_cal, lambda_gc_raw, lambda_gc_final, calibration_mode = _apply_stat_calibration(
        cfg=cfg,
        pvals=pvals,
        chi2_stats=chi2_stats,
        unrelated_scale=unrelated_scale,
        verbose=verbose,
    )
    cost.timing_loco_test_s = time.perf_counter() - t2

    cost.timing_total_s = time.perf_counter() - t_total
    finalize_cost(cost)

    metadata = build_metadata(
        state=fit.state,
        cost=cost,
        h2_hat=h2_hat,
        pi_selected=pi_selected,
        slab_scale_selected=slab_scale_selected,
        sigma_slab2_selected=sigma_slab2_selected,
        prior_tuning_metric=cfg.prior_tune_metric,
        prior_tuning_score=prior_tuning_score,
        prior_stage1=prior_stage1,
        prior_stage2=prior_stage2,
        loco_mode=loco_mode_effective,
        calibration_mode=calibration_mode,
        lambda_gc_raw=lambda_gc_raw,
        lambda_gc_final=lambda_gc_final,
        n_markers_fit=data.m_effective,
    )
    metadata["sigma_spike2_effective"] = _effective_sigma_spike2(cfg, sigma_slab2_selected)
    metadata["sigma_test_source"] = "residual_loco"
    metadata["robust_se_applied"] = bool(cfg.robust_se and cfg.test_method == "wald")
    if cfg.calibrate_stat_scale == "unrelated_subset":
        metadata["unrelated_subset_n"] = int(unrelated_subset.size if unrelated_subset is not None else 0)
        metadata["unrelated_scale_estimate"] = float(unrelated_scale) if np.isfinite(unrelated_scale) else float("nan")

    if verbose:
        print(
            f"BAYESLOCO complete: total={cost.timing_total_s:.2f}s, "
            f"prior_tune={cost.timing_prior_tune_s:.2f}s, fit={cost.timing_main_fit_s:.2f}s, "
            f"test={cost.timing_loco_test_s:.2f}s, pass_equiv={cost.pass_equiv_total:.2f}"
        )

    return AssociationResults(
        effects=effects,
        se=ses,
        pvalues=np.asarray(pvals_cal, dtype=np.float64),
        metadata=metadata,
    )
