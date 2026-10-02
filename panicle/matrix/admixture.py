"""Model-based ancestry estimation (ADMIXTURE / STRUCTURE likelihood).

Implements the binomial admixture model used by ADMIXTURE and FRAPPE,

    g_ij ~ Binomial(2, p_ij),   p_ij = sum_k q_ik f_kj,

fitted by the FRAPPE EM algorithm (Tang et al. 2005) with missing calls
masked out. Ancestry proportions Q are an alternative (or complement) to PCs
as fixed-effect structure covariates ("Q + K" model, Yu et al. 2006).

Like ADMIXTURE itself, the model assumes markers are in approximate linkage
equilibrium, so run it on an LD-pruned marker set (see ``matrix.ld.ld_prune``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..utils.data_types import GenotypeMatrix

_EPS = 1e-9


@dataclass
class AdmixtureResult:
    Q: np.ndarray            # (n_individuals, K) ancestry proportions, rows sum to 1
    F: np.ndarray            # (K, n_markers) ancestral allele frequencies
    log_likelihood: float
    n_iter: int
    converged: bool

    def covariates(self) -> np.ndarray:
        """K-1 Q columns, safe to use alongside an intercept.

        Q rows sum to one, so including all K columns would be collinear with
        the intercept; the last component is dropped.
        """
        return self.Q[:, :-1].copy()


def _load_dosages(genotype: Union[GenotypeMatrix, np.ndarray], marker_indices) -> np.ndarray:
    if marker_indices is None:
        marker_indices = np.arange(
            genotype.n_markers if isinstance(genotype, GenotypeMatrix) else genotype.shape[1]
        )
    idx = np.asarray(marker_indices, dtype=np.int64)
    if isinstance(genotype, GenotypeMatrix):
        G = genotype.get_columns(idx, dtype=np.float64, copy=True)
    else:
        G = np.asarray(genotype[:, idx], dtype=np.float64).copy()
    G[(G == -9) | ~np.isfinite(G)] = np.nan
    return G


def _project(Q: np.ndarray, F: np.ndarray):
    """Map parameters back onto the simplex (Q rows) and (0, 1) (F)."""
    Q = np.clip(Q, _EPS, None)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return Q, np.clip(F, _EPS, 1.0 - _EPS)


def fit_admixture(
    genotype: Union[GenotypeMatrix, np.ndarray],
    K: int,
    *,
    marker_indices: Optional[np.ndarray] = None,
    max_iter: int = 2000,
    tol: float = 1e-6,
    n_restarts: int = 3,
    seed: int = 0,
) -> AdmixtureResult:
    """Fit the admixture model with EM and return the best of ``n_restarts``.

    Args:
        genotype: (n_individuals x n_markers) dosages coded 0/1/2, missing as
            -9 or NaN.
        K: number of ancestral populations (>= 2).
        marker_indices: markers to use (normally the LD-pruned set).
        max_iter: EM iteration cap per restart.
        tol: convergence on relative change of the log-likelihood.
        n_restarts: random restarts; the highest likelihood fit is returned.
        seed: RNG seed.
    """
    if K < 2:
        raise ValueError("K must be >= 2")
    G = _load_dosages(genotype, marker_indices)
    obs = np.isfinite(G)
    G0 = np.where(obs, G, 0.0)
    n, m = G.shape
    if m == 0:
        raise ValueError("No markers supplied for admixture estimation")
    n_obs_per_ind = obs.sum(axis=1).astype(np.float64)
    if np.any(n_obs_per_ind == 0):
        raise ValueError("At least one individual has no observed genotypes")

    G1 = G0                                  # allele-1 counts, 0 where missing
    G2 = np.where(obs, 2.0 - G0, 0.0)        # allele-0 counts, 0 where missing
    denom_q = 2.0 * n_obs_per_ind[:, None]

    def em_step(Q, F):
        P = np.clip(Q @ F, _EPS, 1.0 - _EPS)
        A = G1 / P
        B = G2 / (1.0 - P)
        a_sum = F * (Q.T @ A)
        b_sum = (1.0 - F) * (Q.T @ B)
        Q_new = Q * (A @ F.T + B @ (1.0 - F).T) / denom_q
        F_new = a_sum / np.maximum(a_sum + b_sum, _EPS)
        return _project(Q_new, F_new)

    def loglik(Q, F):
        P = np.clip(Q @ F, _EPS, 1.0 - _EPS)
        return float(np.sum(G1 * np.log(P) + G2 * np.log1p(-P)))

    rng = np.random.default_rng(seed)
    best: Optional[AdmixtureResult] = None
    for _ in range(max(1, n_restarts)):
        Q, F = _project(rng.dirichlet(np.ones(K), size=n), rng.uniform(0.05, 0.95, size=(K, m)))
        ll = loglik(Q, F)
        converged = False
        it = 0
        # SQUAREM-accelerated EM (Varadhan & Roland 2008), with a monotone
        # fallback to the plain EM update whenever extrapolation does not help.
        while it < max_iter:
            Q1, F1 = em_step(Q, F)
            Q2, F2 = em_step(Q1, F1)
            it += 2
            rQ, rF = Q1 - Q, F1 - F
            vQ, vF = Q2 - Q1 - rQ, F2 - F1 - rF
            r_norm = np.sqrt(np.sum(rQ ** 2) + np.sum(rF ** 2))
            v_norm = np.sqrt(np.sum(vQ ** 2) + np.sum(vF ** 2))
            Q_next, F_next = Q2, F2
            ll_new = loglik(Q2, F2)
            if v_norm > 0:
                alpha = min(-1.0, -r_norm / v_norm)
                Qx, Fx = _project(Q - 2 * alpha * rQ + alpha ** 2 * vQ,
                                  F - 2 * alpha * rF + alpha ** 2 * vF)
                Qx, Fx = em_step(Qx, Fx)
                it += 1
                ll_x = loglik(Qx, Fx)
                if ll_x >= ll_new:
                    Q_next, F_next, ll_new = Qx, Fx, ll_x
            Q, F = Q_next, F_next
            if abs(ll_new - ll) <= tol * abs(ll):
                ll = ll_new
                converged = True
                break
            ll = ll_new
        fit = AdmixtureResult(Q=Q, F=F, log_likelihood=ll, n_iter=it, converged=converged)
        if best is None or fit.log_likelihood > best.log_likelihood:
            best = fit
    # Order components by decreasing mean ancestry so output is deterministic.
    order = np.argsort(-best.Q.mean(axis=0), kind="stable")
    best.Q = best.Q[:, order]
    best.F = best.F[order, :]
    return best
