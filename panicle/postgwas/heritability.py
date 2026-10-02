"""SNP heritability by REML on the global genomic relationship matrix.

Fits the MLM null model

    y = X b + g + e,   g ~ N(0, Vg K),   e ~ N(0, Ve I)

with the same fixed covariates as the association scan (intercept, external
covariates, PCs / admixture Q) and reports h2 = Vg / (Vg + Ve).

* K is Gower-scaled (tr(K)/n - mean(K) = 1, Legarra 2016) so Vg is the
  genetic variance of the analysed panel, for outbred and inbred lines alike.
  PANICLE's VanRaden kinship already satisfies this.
* The likelihood is evaluated in the eigenbasis of K, so each evaluation is
  O(n); h2 is found by bounded 1-D optimisation of the profiled REML
  likelihood over [0, 1].
* The standard error comes from the inverse REML Fisher (average)
  information of (Vg, Ve), I_ij = tr(P V_i P V_j) / 2, transformed to h2 by
  the delta method - the same approach as GCTA.
* A likelihood-ratio test of h2 > 0 is reported against the h2 = 0 boundary
  (50:50 mixture of chi2_0 and chi2_1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy import optimize
from scipy.stats import chi2

_H2_MAX = 1.0 - 1e-6


@dataclass
class REMLHeritability:
    h2: float
    h2_se: float
    vg: float
    ve: float
    lrt_p: float
    n: int
    n_fixed: int
    at_boundary: bool

    def to_dict(self):
        return asdict(self)


def gower_scale(K: np.ndarray) -> np.ndarray:
    """Scale K so that tr(K)/n - mean(K) = 1."""
    K = np.asarray(K, dtype=np.float64)
    c = np.trace(K) / K.shape[0] - K.mean()
    if not np.isfinite(c) or c <= 0:
        raise ValueError("Kinship matrix has non-positive Gower scale")
    return K / c


def _design(covariates: Optional[np.ndarray], n: int) -> np.ndarray:
    """Intercept + covariates with collinear columns removed."""
    cols = [np.ones((n, 1))]
    if covariates is not None and np.size(covariates):
        C = np.asarray(covariates, dtype=np.float64)
        cols.append(C.reshape(n, -1))
    X = np.column_stack(cols)
    # Keep a linearly independent subset (pivoted QR).
    from scipy.linalg import qr
    _, R, piv = qr(X, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > diag.max() * 1e-10))
    return X[:, np.sort(piv[:rank])]


def _profiled_reml(h2, yt, Xt, s, n, p):
    """-REML log-likelihood (up to a constant) with sigma^2 profiled out.

    V = sigma^2 * (h2 * s + (1 - h2)) in the eigenbasis of K.
    """
    d = h2 * s + (1.0 - h2)
    d = np.maximum(d, 1e-12)
    w = 1.0 / d
    XtWX = Xt.T @ (Xt * w[:, None])
    XtWy = Xt.T @ (yt * w)
    try:
        L = np.linalg.cholesky(XtWX)
    except np.linalg.LinAlgError:
        return np.inf
    beta = np.linalg.solve(XtWX, XtWy)
    r = yt - Xt @ beta
    rss = float(np.sum(r * r * w))
    dof = n - p
    sigma2 = rss / dof
    if sigma2 <= 0:
        return np.inf
    logdet_v = np.sum(np.log(d))
    logdet_xwx = 2.0 * np.sum(np.log(np.diag(L)))
    return 0.5 * (dof * np.log(sigma2) + logdet_v + logdet_xwx + dof)


def _reml_information(vg, ve, Xt, s):
    """REML Fisher information for (Vg, Ve) in the eigenbasis (V diagonal).

    tr(P A P B) for diagonal A, B with P = D - W M W', D = V^-1,
    W = D X, M = (X' D X)^-1, is evaluated in O(n p^2) without forming P.
    """
    dvec = 1.0 / np.maximum(vg * s + ve, 1e-12)
    W = Xt * dvec[:, None]
    M = np.linalg.inv(Xt.T @ W)
    hdiag = np.einsum("ij,jk,ik->i", W, M, W)    # diag(W M W')

    def tr_pApB(a, b):
        t1 = np.sum(dvec * dvec * a * b)
        t2 = np.sum(dvec * hdiag * a * b)
        WaW = W.T @ (W * a[:, None])
        WbW = W.T @ (W * b[:, None])
        t3 = np.trace(M @ WaW @ M @ WbW)
        return t1 - 2.0 * t2 + t3

    one = np.ones_like(s)
    I = 0.5 * np.array([
        [tr_pApB(s, s), tr_pApB(s, one)],
        [tr_pApB(s, one), tr_pApB(one, one)],
    ])
    return I


def reml_heritability(
    y: np.ndarray,
    K: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    *,
    eigen: Optional[tuple] = None,
) -> REMLHeritability:
    """REML SNP heritability with standard error.

    Args:
        y: phenotype (n,), finite values only.
        K: genomic relationship matrix (n, n) for the same individuals.
        covariates: fixed covariates (n, c) excluding the intercept.
        eigen: optional precomputed (eigenvalues, eigenvectors) of the
            Gower-scaled K.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if not np.all(np.isfinite(y)):
        raise ValueError("Phenotype contains non-finite values")
    X = _design(covariates, n)
    p = X.shape[1]
    if n - p < 3:
        raise ValueError("Too few individuals for REML heritability")
    if eigen is None:
        s, U = np.linalg.eigh(gower_scale(K))
    else:
        s, U = eigen
    s = np.maximum(s, 0.0)
    yt = U.T @ y
    Xt = U.T @ X

    obj = lambda h: _profiled_reml(h, yt, Xt, s, n, p)
    # Coarse grid then bounded refinement guards against a multimodal profile.
    grid = np.linspace(0.0, _H2_MAX, 41)
    vals = np.array([obj(h) for h in grid])
    k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    res = optimize.minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-6})
    h2 = float(res.x) if res.fun <= vals[k] else float(grid[k])
    nll = min(float(res.fun), float(vals[k]))

    # Variance components at the optimum.
    d = h2 * s + (1.0 - h2)
    w = 1.0 / np.maximum(d, 1e-12)
    beta = np.linalg.solve(Xt.T @ (Xt * w[:, None]), Xt.T @ (yt * w))
    r = yt - Xt @ beta
    sigma2 = float(np.sum(r * r * w) / (n - p))
    vg, ve = h2 * sigma2, (1.0 - h2) * sigma2

    # SE via REML information + delta method.
    try:
        cov = np.linalg.inv(_reml_information(vg, ve, Xt, s))
        grad = np.array([ve, -vg]) / (vg + ve) ** 2
        h2_se = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
    except np.linalg.LinAlgError:
        h2_se = float("nan")

    # LRT against h2 = 0 (boundary: 50:50 chi2_0 / chi2_1 mixture).
    stat = max(0.0, 2.0 * (obj(0.0) - nll))
    lrt_p = 0.5 * float(chi2.sf(stat, 1)) if stat > 0 else 1.0

    return REMLHeritability(
        h2=h2, h2_se=h2_se, vg=vg, ve=ve, lrt_p=lrt_p, n=n, n_fixed=p,
        at_boundary=bool(h2 < 1e-4 or h2 > _H2_MAX - 1e-4),
    )
