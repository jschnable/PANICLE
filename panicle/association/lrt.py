
import numpy as np
from scipy import stats, optimize
import warnings
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple

from .mlm import _calculate_neg_ml_likelihood

logger = logging.getLogger(__name__)

_H2_MIN = 1e-3
_H2_MAX = 0.999
_H2_CACHE_SCALE = 1_000_000_000_000


def _clamp_scalar(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _clamp_h2(value: float) -> float:
    return _clamp_scalar(float(value), _H2_MIN, _H2_MAX)


def _sanitize_array(arr: np.ndarray, clip: float = 1e6) -> np.ndarray:
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None and arr.size:
        max_abs = np.max(np.abs(arr))
        if max_abs > clip:
            arr = np.clip(arr, -clip, clip, out=arr)
    return arr


@dataclass
class _ProfileMLState:
    """Cached profile-ML evaluation state at one h2 value."""

    success: bool
    h2: float
    neg_loglik: float
    y_pxy: float
    beta_hat: Optional[np.ndarray]
    x_vix: Optional[np.ndarray]
    grad: Optional[float] = None
    hess: Optional[float] = None


@dataclass
class _BatchProfileState:
    """Profile-ML state for a batch of markers at one h2."""

    success: np.ndarray
    neg_loglik: np.ndarray
    grad: np.ndarray
    y_pxy: np.ndarray
    beta_marker: np.ndarray
    inv_b22: np.ndarray


@dataclass
class _BatchSchurScratch:
    """Reusable workspace for `_profile_ml_state_batch_schur`."""

    wx: np.ndarray
    weighted_g: np.ndarray
    coeff_r0: np.ndarray


def _alloc_batch_schur_scratch(n_samples: int, n_covariates: int, n_markers: int) -> _BatchSchurScratch:
    return _BatchSchurScratch(
        wx=np.empty((n_samples, n_covariates), dtype=np.float64),
        weighted_g=np.empty((n_samples, n_markers), dtype=np.float64),
        coeff_r0=np.empty(n_samples, dtype=np.float64),
    )


@contextmanager
def _suppress_known_matmul_runtime_warnings():
    # Some NumPy + BLAS/Accelerate builds leak FPE RuntimeWarnings from matmul
    # even though downstream finite-value checks handle the affected states.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        yield


def _solve_linear_system(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a linear system with robust fallback for near-singular systems."""
    if mat.shape == (1, 1):
        denom = float(mat[0, 0])
        if np.isfinite(denom) and abs(denom) > 1e-14:
            return rhs / denom
    elif mat.shape == (2, 2):
        a00 = float(mat[0, 0])
        a01 = float(mat[0, 1])
        a10 = float(mat[1, 0])
        a11 = float(mat[1, 1])
        det = a00 * a11 - a01 * a10
        if np.isfinite(det) and abs(det) > 1e-14:
            inv_det = 1.0 / det
            if rhs.ndim == 1:
                x0 = (a11 * float(rhs[0]) - a01 * float(rhs[1])) * inv_det
                x1 = (a00 * float(rhs[1]) - a10 * float(rhs[0])) * inv_det
                return np.array([x0, x1], dtype=np.float64)
            x0 = (a11 * rhs[0, ...] - a01 * rhs[1, ...]) * inv_det
            x1 = (a00 * rhs[1, ...] - a10 * rhs[0, ...]) * inv_det
            return np.stack((x0, x1), axis=0)
    try:
        return np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(mat, rhs, rcond=None)[0]


def _profile_ml_state_impl(
    h2: float,
    y: np.ndarray,
    X: np.ndarray,
    eig_safe: np.ndarray,
    *,
    with_derivatives: bool = False,
    with_hessian: bool = False,
    m_diag: Optional[np.ndarray] = None,
    n_samples: Optional[int] = None,
    trusted_inputs: bool = False,
) -> _ProfileMLState:
    """Shared profile-ML evaluator used by both standard and fast paths."""
    if n_samples is None:
        n_samples = int(y.size)

    # H = h2*D + (1-h2)I = I + h2*(D-I)
    var_diag = h2 * eig_safe + (1.0 - h2)
    if trusted_inputs:
        inv_var = 1.0 / var_diag
        vi_x = inv_var[:, np.newaxis] * X
        x_vix = X.T @ vi_x
    else:
        if np.any(var_diag <= 0) or not np.all(np.isfinite(var_diag)):
            return _ProfileMLState(False, h2, np.inf, 0.0, None, None)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            inv_var = 1.0 / var_diag
            vi_x = inv_var[:, np.newaxis] * X
            x_vix = X.T @ vi_x
        if not np.all(np.isfinite(inv_var)) or not np.all(np.isfinite(x_vix)):
            return _ProfileMLState(False, h2, np.inf, 0.0, None, None)

    rhs_y = vi_x.T @ y
    beta_hat = _solve_linear_system(x_vix, rhs_y)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        vi_y = inv_var * y
        pxy = vi_y - vi_x @ beta_hat
    y_pxy = float(np.dot(y, pxy))
    if not np.isfinite(y_pxy) or y_pxy <= 0:
        return _ProfileMLState(False, h2, np.inf, 0.0, None, None)

    neg_loglik = 0.5 * (np.sum(np.log(var_diag)) + n_samples * np.log(y_pxy / n_samples))
    if not np.isfinite(neg_loglik):
        return _ProfileMLState(False, h2, np.inf, 0.0, None, None)

    if not with_derivatives:
        return _ProfileMLState(True, h2, float(neg_loglik), y_pxy, beta_hat, x_vix)

    if m_diag is None:
        m_diag = eig_safe - 1.0

    # GEMMA-style first derivative term for H(h2)=I+h2*(D-I):
    # dl/dh = -1/2 tr(H^-1 M) + n/2 * (y' P M P y)/(y' P y), where M = D-I
    # Optional second derivative:
    # d2l/dh2 = 1/2 tr(H^-1 M H^-1 M) - n/2 * ((A2*A0-A1^2)/A0^2)
    trace_hinv_m = float(np.dot(m_diag, inv_var))

    # A1 = y' P M P y
    m_pxy = m_diag * pxy
    a1 = float(np.dot(m_pxy, pxy))
    if not np.isfinite(a1):
        return _ProfileMLState(False, h2, np.inf, 0.0, None, None)

    dl_dh = -0.5 * trace_hinv_m + 0.5 * n_samples * (a1 / y_pxy)

    # We minimize negative log-likelihood.
    grad = -dl_dh
    hess: Optional[float] = None
    if not np.isfinite(grad):
        return _ProfileMLState(False, h2, np.inf, 0.0, None, None)

    if with_hessian:
        trace_hinv_m_hinv_m = float(np.dot(m_diag * inv_var, m_diag * inv_var))

        # A2 = y' P M P M P y = (M P y)' P (M P y)
        rhs_m = vi_x.T @ m_pxy
        beta_m = _solve_linear_system(x_vix, rhs_m)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p_m_pxy = (inv_var * m_pxy) - vi_x @ beta_m
        a2 = float(np.dot(m_pxy, p_m_pxy))
        if not np.isfinite(a2):
            return _ProfileMLState(False, h2, np.inf, 0.0, None, None)

        d2l_dh2 = 0.5 * trace_hinv_m_hinv_m - 0.5 * n_samples * ((a2 * y_pxy - a1 * a1) / (y_pxy * y_pxy))
        hess = -d2l_dh2
        if not np.isfinite(hess):
            return _ProfileMLState(False, h2, np.inf, 0.0, None, None)

    return _ProfileMLState(
        True,
        h2,
        float(neg_loglik),
        y_pxy,
        beta_hat,
        x_vix,
        float(grad),
        float(hess) if hess is not None else None,
    )


def _profile_ml_state(
    h2: float,
    y: np.ndarray,
    X: np.ndarray,
    eigenvals: np.ndarray,
    *,
    with_derivatives: bool = False,
    with_hessian: bool = False,
) -> _ProfileMLState:
    """Evaluate profile ML objective (and optionally derivatives) at h2.

    Derivative formulas follow GEMMA's exact LMM derivatives for a linear
    covariance parameterization. Here, H(h2) = h2*D + (1-h2)I, so dH/dh2 = D-I.
    """
    if not np.isfinite(h2):
        return _ProfileMLState(False, float(h2), np.inf, 0.0, None, None)

    eig_safe = np.maximum(np.asarray(eigenvals, dtype=np.float64), 1e-6)
    h2 = _clamp_h2(h2)
    y_local = np.asarray(y, dtype=np.float64)
    X_local = np.asarray(X, dtype=np.float64)
    return _profile_ml_state_impl(
        h2,
        y_local,
        X_local,
        eig_safe,
        with_derivatives=with_derivatives,
        with_hessian=with_hessian,
        m_diag=(eig_safe - 1.0) if with_derivatives else None,
        n_samples=int(y_local.size),
        trusted_inputs=False,
    )


def _optimize_alt_h2_brent(
    y: np.ndarray,
    x_alt: np.ndarray,
    eigenvals: np.ndarray,
) -> _ProfileMLState:
    """Reference bounded Brent optimizer (legacy path)."""

    def alt_neg_ml_likelihood(h2: float) -> float:
        return _calculate_neg_ml_likelihood(h2, y, x_alt, eigenvals)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize.minimize_scalar(
            alt_neg_ml_likelihood,
            bounds=(0.001, 0.999),
            method="bounded",
            options={"xatol": 1.22e-4, "maxiter": 100},
        )

    if not result.success:
        return _ProfileMLState(False, 0.5, np.inf, 0.0, None, None)

    return _profile_ml_state(float(result.x), y, x_alt, eigenvals, with_derivatives=False)


def _optimize_alt_h2_gemma_newton(
    y: np.ndarray,
    x_alt: np.ndarray,
    eigenvals: np.ndarray,
    *,
    h2_init: Optional[float] = None,
    # Keep Newton polishing opt-in. Root bracketing is usually sufficient and
    # enabling Newton by default can add overhead for large marker scans.
    max_iter: int = 0,
    tol: float = 1e-5,
) -> _ProfileMLState:
    """GEMMA-style profile-ML optimizer.

    Mirrors GEMMA's `CalcLambda` strategy:
    1) log-spaced scan for derivative sign changes
    2) Brent solve on each bracketing interval
    3) boundary likelihood comparison when no interior root exists
    """
    h2_min = _H2_MIN
    h2_max = _H2_MAX
    n_regions = 10
    lambda_min = h2_min / (1.0 - h2_min)
    lambda_max = h2_max / (1.0 - h2_max)
    lambda_tol = 1e-4
    root_max_iter = 24

    y_local = np.asarray(y, dtype=np.float64)
    X_local = np.asarray(x_alt, dtype=np.float64)
    eig_safe = np.maximum(np.asarray(eigenvals, dtype=np.float64), 1e-6)
    m_diag = eig_safe - 1.0
    n_samples = y_local.size
    n_covariates = int(X_local.shape[1])
    use_uab_fast = n_covariates <= 12

    if use_uab_fast:
        # Precompute X-only terms used in weighted normal equations:
        # X'V^-1X and X'V^-1y become weighted sums over these arrays.
        xx_terms = np.einsum("ni,nj->ijn", X_local, X_local, optimize=True)
        xy_terms = X_local * y_local[:, np.newaxis]
        x_t = X_local.T
    else:
        xx_terms = None
        xy_terms = None
        x_t = None

    def _lambda_from_h2(h2: float) -> float:
        h2 = _clamp_h2(h2)
        return h2 / max(1e-12, 1.0 - h2)

    def _h2_from_lambda(lam: float) -> float:
        lam = _clamp_scalar(float(lam), lambda_min, lambda_max)
        return lam / (1.0 + lam)

    state_cache: dict[int, _ProfileMLState] = {}
    state_hess_cache: dict[int, _ProfileMLState] = {}

    def _profile_ml_state_fast(h2: float, *, with_hessian: bool) -> _ProfileMLState:
        """Fast-path profile ML state evaluator for finite, sanitized inputs."""
        h2 = _clamp_h2(h2)
        try:
            if use_uab_fast and xx_terms is not None and xy_terms is not None and x_t is not None:
                var_diag = h2 * eig_safe + (1.0 - h2)
                inv_var = 1.0 / var_diag

                with _suppress_known_matmul_runtime_warnings():
                    x_vix = np.tensordot(inv_var, xx_terms, axes=(0, 2))
                    rhs_y = xy_terms.T @ inv_var
                    beta_hat = _solve_linear_system(x_vix, rhs_y)
                    resid = y_local - (X_local @ beta_hat)
                pxy = inv_var * resid
                y_pxy = float(np.dot(y_local, pxy))
                if (not np.isfinite(y_pxy)) or y_pxy <= 0.0:
                    return _ProfileMLState(False, h2, np.inf, 0.0, None, None)

                neg_loglik = 0.5 * (np.sum(np.log(var_diag)) + n_samples * np.log(y_pxy / n_samples))
                if not np.isfinite(neg_loglik):
                    return _ProfileMLState(False, h2, np.inf, 0.0, None, None)

                trace_hinv_m = float(np.dot(m_diag, inv_var))
                m_pxy = m_diag * pxy
                a1 = float(np.dot(m_pxy, pxy))
                if not np.isfinite(a1):
                    return _ProfileMLState(False, h2, np.inf, 0.0, None, None)
                grad = -(-0.5 * trace_hinv_m + 0.5 * n_samples * (a1 / y_pxy))
                if not np.isfinite(grad):
                    return _ProfileMLState(False, h2, np.inf, 0.0, None, None)

                hess: Optional[float] = None
                if with_hessian:
                    trace_hinv_m_hinv_m = float(np.dot(m_diag * inv_var, m_diag * inv_var))
                    with _suppress_known_matmul_runtime_warnings():
                        rhs_m = x_t @ (inv_var * m_pxy)
                        beta_m = _solve_linear_system(x_vix, rhs_m)
                        p_m_pxy = inv_var * (m_pxy - (X_local @ beta_m))
                    a2 = float(np.dot(m_pxy, p_m_pxy))
                    if not np.isfinite(a2):
                        return _ProfileMLState(False, h2, np.inf, 0.0, None, None)
                    hess_val = -(0.5 * trace_hinv_m_hinv_m - 0.5 * n_samples * ((a2 * y_pxy - a1 * a1) / (y_pxy * y_pxy)))
                    if not np.isfinite(hess_val):
                        return _ProfileMLState(False, h2, np.inf, 0.0, None, None)
                    hess = float(hess_val)

                return _ProfileMLState(
                    True,
                    h2,
                    float(neg_loglik),
                    y_pxy,
                    beta_hat,
                    x_vix,
                    float(grad),
                    hess,
                )
            return _profile_ml_state_impl(
                h2,
                y_local,
                X_local,
                eig_safe,
                with_derivatives=True,
                with_hessian=with_hessian,
                m_diag=m_diag,
                n_samples=n_samples,
                trusted_inputs=True,
            )
        except Exception:
            logger.debug("GEMMA fast-path profile state failed; falling back", exc_info=True)
            return _ProfileMLState(False, h2, np.inf, 0.0, None, None)

    def _cache_key(h2: float) -> int:
        return int(h2 * _H2_CACHE_SCALE + 0.5)

    def _state_grad(h2: float) -> _ProfileMLState:
        h2 = _clamp_h2(h2)
        key = _cache_key(h2)
        cached = state_cache.get(key)
        if cached is not None:
            return cached
        state = _profile_ml_state_fast(h2, with_hessian=False)
        if (
            (not state.success)
            or (state.grad is None)
            or (not np.isfinite(state.grad))
            or (not np.isfinite(state.neg_loglik))
        ):
            state = _profile_ml_state(
                h2,
                y,
                x_alt,
                eigenvals,
                with_derivatives=True,
                with_hessian=False,
            )
        state_cache[key] = state
        return state

    def _state_hess(h2: float) -> _ProfileMLState:
        h2 = _clamp_h2(h2)
        key = _cache_key(h2)
        cached = state_hess_cache.get(key)
        if cached is not None:
            return cached
        state = _profile_ml_state_fast(h2, with_hessian=True)
        if (
            (not state.success)
            or (state.grad is None)
            or (not np.isfinite(state.grad))
            or (state.hess is None)
            or (not np.isfinite(state.hess))
            or (not np.isfinite(state.neg_loglik))
        ):
            state = _profile_ml_state(
                h2,
                y,
                x_alt,
                eigenvals,
                with_derivatives=True,
                with_hessian=True,
            )
        state_hess_cache[key] = state
        state_cache.setdefault(key, state)
        return state

    def _valid(state: _ProfileMLState) -> bool:
        return (
            state.success
            and np.isfinite(state.neg_loglik)
            and (state.grad is not None)
            and np.isfinite(state.grad)
        )

    def _grad_at_lambda(lam: float) -> float:
        state = _state_grad(_h2_from_lambda(lam))
        if not _valid(state):
            return np.nan
        return float(state.grad)

    def _solve_root_in_bracket(
        lam_l: float,
        lam_h: float,
        grad_l: float,
        grad_h: float,
    ) -> Optional[float]:
        """Safeguarded secant/false-position root solver on a known sign-change bracket."""
        a = float(lam_l)
        b = float(lam_h)
        fa = float(grad_l)
        fb = float(grad_h)

        if not np.isfinite(fa) or not np.isfinite(fb):
            return None
        if abs(fa) <= tol:
            return a
        if abs(fb) <= tol:
            return b
        if fa * fb > 0.0:
            return None

        for _ in range(root_max_iter):
            width = b - a
            if width <= max(lambda_tol, lambda_tol * max(abs(a), abs(b), 1.0)):
                return 0.5 * (a + b)

            denom = fb - fa
            if np.isfinite(denom) and abs(denom) > 1e-18:
                c = (a * fb - b * fa) / denom
            else:
                c = 0.5 * (a + b)
            if (not np.isfinite(c)) or c <= a or c >= b:
                c = 0.5 * (a + b)

            fc = _grad_at_lambda(c)
            if not np.isfinite(fc):
                c = 0.5 * (a + b)
                fc = _grad_at_lambda(c)
                if not np.isfinite(fc):
                    return None

            if abs(fc) <= tol:
                return c

            if fa * fc < 0.0:
                b = c
                fb = fc
                fa *= 0.5
            else:
                a = c
                fa = fc
                fb *= 0.5

        return 0.5 * (a + b)

    candidate_states: list[_ProfileMLState] = []
    root_brackets: list[tuple[float, float, float, float]] = []

    def _append_bracket(
        lam_l: float,
        lam_h: float,
        grad_l: float,
        grad_h: float,
    ) -> None:
        if (not np.isfinite(grad_l)) or (not np.isfinite(grad_h)):
            return
        if lam_h - lam_l <= 1e-12:
            return
        if grad_l * grad_h < 0.0:
            root_brackets.append((float(lam_l), float(lam_h), float(grad_l), float(grad_h)))

    def _scan_for_brackets() -> None:
        lambda_edges = np.geomspace(lambda_min, lambda_max, num=n_regions + 1)
        n_edges = int(lambda_edges.size)
        grad_edges = np.empty(n_edges, dtype=np.float64)
        valid_edges = np.zeros(n_edges, dtype=bool)

        # Evaluate each edge once and reuse for adjacent intervals.
        for i in range(n_edges):
            lam = float(lambda_edges[i])
            state = _state_grad(_h2_from_lambda(lam))
            if not _valid(state):
                grad_edges[i] = np.nan
                continue
            grad_val = float(state.grad)
            grad_edges[i] = grad_val
            valid_edges[i] = True
            if abs(grad_val) <= tol:
                candidate_states.append(state)

        for i in range(n_regions):
            if not (valid_edges[i] and valid_edges[i + 1]):
                continue
            _append_bracket(
                float(lambda_edges[i]),
                float(lambda_edges[i + 1]),
                float(grad_edges[i]),
                float(grad_edges[i + 1]),
            )

    # Seed with boundaries and a user-supplied initial guess.
    state_low = _state_grad(h2_min)
    state_high = _state_grad(h2_max)
    if _valid(state_low):
        candidate_states.append(state_low)
    if _valid(state_high):
        candidate_states.append(state_high)

    state_init: Optional[_ProfileMLState] = None
    if h2_init is not None and np.isfinite(h2_init):
        state_init = _state_grad(float(h2_init))
        if _valid(state_init):
            candidate_states.append(state_init)

    # Fast path: try null-centered bracketing first; fallback to full scan.
    used_null_centered = False
    if (
        state_init is not None
        and _valid(state_init)
        and _valid(state_low)
        and _valid(state_high)
    ):
        lam_init = _lambda_from_h2(state_init.h2)
        grad_init = float(state_init.grad)
        grad_low = float(state_low.grad)
        grad_high = float(state_high.grad)

        before_n = len(root_brackets)
        _append_bracket(lambda_min, lam_init, grad_low, grad_init)
        _append_bracket(lam_init, lambda_max, grad_init, grad_high)
        used_null_centered = len(root_brackets) > before_n

    if not used_null_centered:
        _scan_for_brackets()

    for lam_l, lam_h, grad_l, grad_h in root_brackets:
        lam_root = _solve_root_in_bracket(lam_l, lam_h, grad_l, grad_h)
        if lam_root is None:
            # Conservative fallback to SciPy Brent for rare hard brackets.
            try:
                lam_root = optimize.brentq(
                    _grad_at_lambda,
                    lam_l,
                    lam_h,
                    xtol=1e-4,
                    rtol=1e-4,
                    maxiter=100,
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue

        h2_root = _h2_from_lambda(lam_root)
        state_root = _state_grad(h2_root)
        if not _valid(state_root):
            continue

        # Optional Newton polishing, but never fail the whole solve on this step.
        state_best_local = state_root
        for _ in range(max(0, min(max_iter, 8))):
            state_cur = _state_hess(state_best_local.h2)
            if not _valid(state_cur):
                break
            if abs(float(state_cur.grad)) <= tol:
                state_best_local = state_cur
                break
            if state_cur.hess is None or (not np.isfinite(state_cur.hess)) or abs(state_cur.hess) < 1e-12:
                break

            step = -float(state_cur.grad) / float(state_cur.hess)
            if not np.isfinite(step):
                break
            step = _clamp_scalar(step, -0.2, 0.2)

            accepted = False
            trial_step = step
            for _ in range(8):
                h2_trial = _clamp_scalar(state_cur.h2 + trial_step, h2_min, h2_max)
                if abs(h2_trial - state_cur.h2) <= tol:
                    break
                state_trial = _state_grad(h2_trial)
                if _valid(state_trial) and state_trial.neg_loglik <= state_best_local.neg_loglik + 1e-12:
                    state_best_local = state_trial
                    accepted = True
                    break
                trial_step *= 0.5
            if not accepted:
                break

        candidate_states.append(state_best_local)

    if not candidate_states:
        return _ProfileMLState(False, 0.5, np.inf, 0.0, None, None)

    best_state = min(candidate_states, key=lambda s: s.neg_loglik)
    if not np.isfinite(best_state.neg_loglik):
        return _ProfileMLState(False, best_state.h2, np.inf, 0.0, None, None)
    return best_state


def _normalize_lrt_solver(solver: str) -> str:
    solver_norm = str(solver).strip().upper()
    if solver_norm not in {"GEMMA", "BRENT", "AUTO"}:
        raise ValueError(f"Unknown LRT solver: {solver}")
    return solver_norm


def _fit_marker_lrt_core(
    y_transformed: np.ndarray,
    X_alt: np.ndarray,
    eigenvals: np.ndarray,
    null_neg_loglik: float,
    *,
    null_h2: Optional[float],
    solver_norm: str,
) -> Tuple[float, float, float, float]:
    # Fast path: GEMMA-style derivative optimizer with automatic fallback to Brent.
    alt_state = _ProfileMLState(False, 0.5, np.inf, 0.0, None, None)
    if solver_norm == "GEMMA":
        try:
            alt_state = _optimize_alt_h2_gemma_newton(
                y_transformed,
                X_alt,
                eigenvals,
                h2_init=null_h2,
            )
        except Exception:
            alt_state = _ProfileMLState(False, 0.5, np.inf, 0.0, None, None)

    if (solver_norm == "BRENT") or (not alt_state.success):
        try:
            alt_state = _optimize_alt_h2_brent(y_transformed, X_alt, eigenvals)
        except Exception:
            return 0.0, 1.0, 0.0, float("inf")

    if not alt_state.success or not np.isfinite(alt_state.neg_loglik):
        return 0.0, 1.0, 0.0, float("inf")

    # LRT = 2 * (LL_alt - LL_null) = 2 * (null_neg - alt_neg) for negative LLs.
    lrt_stat = 2.0 * (float(null_neg_loglik) - float(alt_state.neg_loglik))
    if not np.isfinite(lrt_stat) or lrt_stat < 0:
        lrt_stat = 0.0
    p_value = float(stats.chi2.sf(lrt_stat, df=1))

    if alt_state.beta_hat is None or alt_state.x_vix is None:
        return float(lrt_stat), p_value, 0.0, float("inf")

    beta_marker = float(alt_state.beta_hat[-1])

    # SE from covariance matrix diag, scaled by ML variance estimate.
    df = len(y_transformed) - X_alt.shape[1]
    v_base = alt_state.y_pxy / max(1, df)
    if not np.isfinite(v_base) or v_base < 0:
        return float(lrt_stat), p_value, beta_marker, float("inf")

    e_last = np.zeros(alt_state.x_vix.shape[0], dtype=np.float64)
    e_last[-1] = 1.0
    inv_col_last = _solve_linear_system(alt_state.x_vix, e_last)
    var_marker = float(v_base * inv_col_last[-1])
    se_marker = float(np.sqrt(max(var_marker, 0.0)))

    return float(lrt_stat), p_value, beta_marker, se_marker


def _profile_ml_state_batch_schur(
    h2: float,
    y: np.ndarray,
    X: np.ndarray,
    G: np.ndarray,
    eig_safe: np.ndarray,
    m_diag: np.ndarray,
    scratch: Optional[_BatchSchurScratch] = None,
) -> _BatchProfileState:
    """Batch profile-ML evaluation for markers sharing the same null model."""
    n_samples = int(y.size)
    n_markers = int(G.shape[1])
    if n_markers == 0:
        empty = np.empty(0, dtype=np.float64)
        return _BatchProfileState(
            success=np.zeros(0, dtype=bool),
            neg_loglik=empty,
            grad=empty,
            y_pxy=empty,
            beta_marker=empty,
            inv_b22=empty,
        )

    h2 = _clamp_h2(h2)
    var_diag = h2 * eig_safe + (1.0 - h2)
    if np.any(var_diag <= 0.0) or not np.all(np.isfinite(var_diag)):
        bad = np.full(n_markers, np.nan, dtype=np.float64)
        return _BatchProfileState(
            success=np.zeros(n_markers, dtype=bool),
            neg_loglik=bad.copy(),
            grad=bad.copy(),
            y_pxy=bad.copy(),
            beta_marker=bad.copy(),
            inv_b22=bad.copy(),
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv_var = 1.0 / var_diag

    if (
        scratch is not None
        and scratch.wx.shape == X.shape
        and scratch.weighted_g.shape == G.shape
        and scratch.coeff_r0.shape == y.shape
    ):
        wx = scratch.wx
        weighted_g = scratch.weighted_g
        coeff_r0 = scratch.coeff_r0
    else:
        wx = np.empty_like(X, dtype=np.float64)
        weighted_g = np.empty_like(G, dtype=np.float64)
        coeff_r0 = np.empty_like(y, dtype=np.float64)

    with _suppress_known_matmul_runtime_warnings():
        np.multiply(inv_var[:, np.newaxis], X, out=wx)
        A = X.T @ wx
        q = wx.T @ y
        wy = inv_var * y
    y_wy = float(np.dot(y, wy))
    sum_log_var = float(np.sum(np.log(var_diag)))

    try:
        a_inv_q = _solve_linear_system(A, q)
        with _suppress_known_matmul_runtime_warnings():
            np.multiply(inv_var[:, np.newaxis], G, out=weighted_g)
            B = X.T @ weighted_g
        a_inv_B = _solve_linear_system(A, B)
    except Exception:
        bad = np.full(n_markers, np.nan, dtype=np.float64)
        return _BatchProfileState(
            success=np.zeros(n_markers, dtype=bool),
            neg_loglik=bad.copy(),
            grad=bad.copy(),
            y_pxy=bad.copy(),
            beta_marker=bad.copy(),
            inv_b22=bad.copy(),
        )

    c = np.sum(G * weighted_g, axis=0)
    with _suppress_known_matmul_runtime_warnings():
        u = G.T @ wy

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        S = c - np.sum(B * a_inv_B, axis=0)
        inv_b22 = np.where(np.abs(S) > 1e-12, 1.0 / S, np.nan)
        beta_marker = u - np.sum(B * a_inv_q[:, np.newaxis], axis=0)
        beta_marker = beta_marker * inv_b22

    y_base = y_wy - float(np.dot(q, a_inv_q))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        numer = u - np.sum(B * a_inv_q[:, np.newaxis], axis=0)
        y_pxy = y_base - numer * numer * inv_b22
        neg_loglik = 0.5 * (sum_log_var + n_samples * np.log(y_pxy / n_samples))

    coeff = m_diag * (inv_var * inv_var)
    # Gradient term: y'PMPy using Schur-decomposed moments to avoid building
    # the full (n x m) residual matrix.
    with _suppress_known_matmul_runtime_warnings():
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r0 = y - (X @ a_inv_q)
            np.multiply(coeff, r0, out=coeff_r0)
            c0 = float(np.dot(r0, coeff_r0))

            # Reuse weighted genotype buffer for coeff-weighted terms.
            np.multiply(coeff[:, np.newaxis], G, out=weighted_g)
            d = weighted_g.T @ r0
            k = X.T @ coeff_r0
            c1 = d - np.sum(a_inv_B * k[:, np.newaxis], axis=0)

            g2 = np.sum(G * weighted_g, axis=0)
            xcg = X.T @ weighted_g
            term2 = np.sum(a_inv_B * xcg, axis=0)
            xcx = X.T @ (coeff[:, np.newaxis] * X)
            term3 = np.sum(a_inv_B * (xcx @ a_inv_B), axis=0)
            c2 = g2 - (2.0 * term2) + term3

    beta_sq = beta_marker * beta_marker
    a1 = c0 - (2.0 * beta_marker * c1) + (beta_sq * c2)
    trace_hinv_m = float(np.dot(m_diag, inv_var))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dl_dh = -0.5 * trace_hinv_m + 0.5 * n_samples * (a1 / y_pxy)
    grad = -dl_dh

    success = (
        np.isfinite(S)
        & (S > 1e-12)
        & np.isfinite(inv_b22)
        & np.isfinite(beta_marker)
        & np.isfinite(y_pxy)
        & (y_pxy > 0.0)
        & np.isfinite(neg_loglik)
        & np.isfinite(grad)
    )
    return _BatchProfileState(
        success=success,
        neg_loglik=np.asarray(neg_loglik, dtype=np.float64),
        grad=np.asarray(grad, dtype=np.float64),
        y_pxy=np.asarray(y_pxy, dtype=np.float64),
        beta_marker=np.asarray(beta_marker, dtype=np.float64),
        inv_b22=np.asarray(inv_b22, dtype=np.float64),
    )


def _optimize_alt_h2_from_bracket(
    y: np.ndarray,
    x_alt: np.ndarray,
    eigenvals: np.ndarray,
    *,
    lam_l: float,
    lam_h: float,
    grad_l: float,
    grad_h: float,
    tol: float = 1e-5,
) -> _ProfileMLState:
    """Root-solve GEMMA derivative within a known sign-change bracket."""
    lambda_min = _H2_MIN / (1.0 - _H2_MIN)
    lambda_max = _H2_MAX / (1.0 - _H2_MAX)
    lambda_tol = 1e-4
    root_max_iter = 24

    eig_local = np.asarray(eigenvals, dtype=np.float64)
    y_local = np.asarray(y, dtype=np.float64)
    x_local = np.asarray(x_alt, dtype=np.float64)

    def _h2_from_lambda(lam: float) -> float:
        lam = _clamp_scalar(float(lam), lambda_min, lambda_max)
        return lam / (1.0 + lam)

    def _state_grad(h2: float) -> _ProfileMLState:
        return _profile_ml_state(
            h2,
            y_local,
            x_local,
            eig_local,
            with_derivatives=True,
            with_hessian=False,
        )

    def _state_at_lambda(lam: float) -> _ProfileMLState:
        return _state_grad(_h2_from_lambda(lam))

    a = float(lam_l)
    b = float(lam_h)
    fa = float(grad_l)
    fb = float(grad_h)
    if (not np.isfinite(fa)) or (not np.isfinite(fb)) or (fa * fb > 0.0):
        return _ProfileMLState(False, 0.5, np.inf, 0.0, None, None)

    state_a = _state_at_lambda(a)
    state_b = _state_at_lambda(b)
    candidate_states: list[_ProfileMLState] = []
    if state_a.success:
        candidate_states.append(state_a)
    if state_b.success:
        candidate_states.append(state_b)

    if abs(fa) <= tol:
        return state_a if state_a.success else _ProfileMLState(False, 0.5, np.inf, 0.0, None, None)
    if abs(fb) <= tol:
        return state_b if state_b.success else _ProfileMLState(False, 0.5, np.inf, 0.0, None, None)

    lam_root: Optional[float] = None
    for _ in range(root_max_iter):
        width = b - a
        if width <= max(lambda_tol, lambda_tol * max(abs(a), abs(b), 1.0)):
            lam_root = 0.5 * (a + b)
            break

        denom = fb - fa
        if np.isfinite(denom) and abs(denom) > 1e-18:
            c = (a * fb - b * fa) / denom
        else:
            c = 0.5 * (a + b)
        if (not np.isfinite(c)) or c <= a or c >= b:
            c = 0.5 * (a + b)

        state_c = _state_at_lambda(c)
        if not state_c.success or (state_c.grad is None) or (not np.isfinite(state_c.grad)):
            c = 0.5 * (a + b)
            state_c = _state_at_lambda(c)
            if not state_c.success or (state_c.grad is None) or (not np.isfinite(state_c.grad)):
                break
        candidate_states.append(state_c)
        fc = float(state_c.grad)

        if abs(fc) <= tol:
            lam_root = c
            break
        if fa * fc < 0.0:
            b = c
            fb = fc
            fa *= 0.5
        else:
            a = c
            fa = fc
            fb *= 0.5
    else:
        lam_root = 0.5 * (a + b)

    if lam_root is None:
        try:
            lam_root = optimize.brentq(
                lambda lam: float(_state_at_lambda(lam).grad),
                a,
                b,
                xtol=1e-4,
                rtol=1e-4,
                maxiter=100,
            )
        except Exception:
            if candidate_states:
                return min(candidate_states, key=lambda s: s.neg_loglik if np.isfinite(s.neg_loglik) else np.inf)
            return _ProfileMLState(False, 0.5, np.inf, 0.0, None, None)

    state_root = _state_at_lambda(float(lam_root))
    if state_root.success:
        candidate_states.append(state_root)
    if not candidate_states:
        return _ProfileMLState(False, 0.5, np.inf, 0.0, None, None)
    best_state = min(candidate_states, key=lambda s: s.neg_loglik if np.isfinite(s.neg_loglik) else np.inf)
    if not best_state.success or not np.isfinite(best_state.neg_loglik):
        return _ProfileMLState(False, 0.5, np.inf, 0.0, None, None)
    return best_state


def fit_markers_lrt_batch_prebuilt(
    y_transformed: np.ndarray,
    X_transformed: np.ndarray,
    G_transformed: np.ndarray,
    eigenvals: np.ndarray,
    null_neg_loglik: float,
    *,
    null_h2: Optional[float] = None,
    solver: str = "GEMMA",
    assume_sanitized: bool = False,
    solver_norm: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch LRT for markers that share y/X/eigen null model context."""
    if assume_sanitized:
        y_local = y_transformed
        X_local = X_transformed
        G_local = G_transformed
    else:
        y_local = _sanitize_array(y_transformed)
        X_local = _sanitize_array(X_transformed)
        G_local = _sanitize_array(G_transformed)

    y_local = np.asarray(y_local, dtype=np.float64)
    X_local = np.asarray(X_local, dtype=np.float64)
    G_local = np.asarray(G_local, dtype=np.float64)
    if G_local.ndim == 1:
        G_local = G_local[:, np.newaxis]

    n_samples = int(y_local.size)
    n_markers = int(G_local.shape[1])
    p_values = np.ones(n_markers, dtype=np.float64)
    betas = np.zeros(n_markers, dtype=np.float64)
    std_errors = np.full(n_markers, np.inf, dtype=np.float64)
    if n_markers == 0:
        return p_values, betas, std_errors

    solver_key = solver_norm if solver_norm is not None else _normalize_lrt_solver(solver)
    if solver_key == "AUTO":
        solver_key = "BRENT" if n_samples < 250 else "GEMMA"

    # Non-GEMMA paths and tiny batches use the existing per-marker exact solver.
    if solver_key != "GEMMA" or n_markers < 8:
        X_alt = np.empty((X_local.shape[0], X_local.shape[1] + 1), dtype=np.float64)
        X_alt[:, :-1] = X_local
        for j in range(n_markers):
            X_alt[:, -1] = G_local[:, j]
            _, p_val, beta, se = fit_marker_lrt_prebuilt(
                y_local,
                X_alt,
                eigenvals,
                null_neg_loglik,
                null_h2=null_h2,
                solver_norm=solver_key,
                assume_sanitized=True,
            )
            p_values[j] = p_val
            betas[j] = beta
            std_errors[j] = se
        return p_values, betas, std_errors

    eig_safe = np.maximum(np.asarray(eigenvals, dtype=np.float64), 1e-6)
    m_diag = eig_safe - 1.0
    lambda_min = _H2_MIN / (1.0 - _H2_MIN)
    lambda_max = _H2_MAX / (1.0 - _H2_MAX)
    tol = 1e-5
    n_regions = 10
    # Shared Schur workspace across h2 evaluations avoids repeated large
    # temporary allocations for weighted genotype/covariate buffers.
    schur_scratch = _alloc_batch_schur_scratch(n_samples, X_local.shape[1], n_markers)

    lambda_edges = np.geomspace(lambda_min, lambda_max, num=n_regions + 1)
    h2_edges = lambda_edges / (1.0 + lambda_edges)
    edge_states = [
        _profile_ml_state_batch_schur(
            float(h2),
            y_local,
            X_local,
            G_local,
            eig_safe,
            m_diag,
            scratch=schur_scratch,
        )
        for h2 in h2_edges
    ]
    state_low = edge_states[0]
    state_high = edge_states[-1]

    state_init: Optional[_BatchProfileState] = None
    lambda_init: Optional[float] = None
    if null_h2 is not None and np.isfinite(null_h2):
        h2_init = _clamp_h2(float(null_h2))
        lambda_init = h2_init / max(1e-12, 1.0 - h2_init)
        state_init = _profile_ml_state_batch_schur(
            h2_init,
            y_local,
            X_local,
            G_local,
            eig_safe,
            m_diag,
            scratch=schur_scratch,
        )

    # Track best finite state seen from shared evaluations.
    best_neg = np.full(n_markers, np.inf, dtype=np.float64)
    best_h2 = np.full(n_markers, np.nan, dtype=np.float64)
    best_beta = np.zeros(n_markers, dtype=np.float64)
    best_y_pxy = np.full(n_markers, np.nan, dtype=np.float64)
    best_inv_b22 = np.full(n_markers, np.nan, dtype=np.float64)
    best_valid = np.zeros(n_markers, dtype=bool)

    def _consider_state(state: _BatchProfileState, h2_value: float) -> None:
        valid = state.success & np.isfinite(state.neg_loglik)
        take = valid & ((~best_valid) | (state.neg_loglik < best_neg))
        if not np.any(take):
            return
        best_valid[take] = True
        best_neg[take] = state.neg_loglik[take]
        best_h2[take] = h2_value
        best_beta[take] = state.beta_marker[take]
        best_y_pxy[take] = state.y_pxy[take]
        best_inv_b22[take] = state.inv_b22[take]

    _consider_state(state_low, _H2_MIN)
    _consider_state(state_high, _H2_MAX)
    for edge_h2, edge_state in zip(h2_edges, edge_states):
        _consider_state(edge_state, float(edge_h2))
    if state_init is not None:
        _consider_state(state_init, _clamp_h2(float(null_h2)))

    bracket_count = np.zeros(n_markers, dtype=np.int32)
    first_lam_l = np.full(n_markers, np.nan, dtype=np.float64)
    first_lam_h = np.full(n_markers, np.nan, dtype=np.float64)
    first_grad_l = np.full(n_markers, np.nan, dtype=np.float64)
    first_grad_h = np.full(n_markers, np.nan, dtype=np.float64)

    def _record_brackets(mask: np.ndarray, lam_l: float, lam_h: float, grad_l: np.ndarray, grad_h: np.ndarray) -> None:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return
        first_mask = bracket_count[idx] == 0
        if np.any(first_mask):
            first_idx = idx[first_mask]
            first_lam_l[first_idx] = float(lam_l)
            first_lam_h[first_idx] = float(lam_h)
            first_grad_l[first_idx] = grad_l[first_idx]
            first_grad_h[first_idx] = grad_h[first_idx]
        bracket_count[idx] += 1

    if state_init is not None and lambda_init is not None:
        grad_low = state_low.grad
        grad_high = state_high.grad
        grad_init = state_init.grad
        left_mask = (
            state_low.success
            & state_init.success
            & np.isfinite(grad_low)
            & np.isfinite(grad_init)
            & (grad_low * grad_init < 0.0)
        )
        right_mask = (
            state_init.success
            & state_high.success
            & np.isfinite(grad_init)
            & np.isfinite(grad_high)
            & (grad_init * grad_high < 0.0)
        )
        _record_brackets(left_mask, lambda_min, float(lambda_init), grad_low, grad_init)
        _record_brackets(right_mask, float(lambda_init), lambda_max, grad_init, grad_high)

    for i in range(n_regions):
        scan_mask = bracket_count == 0
        if not np.any(scan_mask):
            break
        st_l = edge_states[i]
        st_h = edge_states[i + 1]
        grad_l = st_l.grad
        grad_h = st_h.grad
        mask = (
            scan_mask
            & st_l.success
            & st_h.success
            & np.isfinite(grad_l)
            & np.isfinite(grad_h)
            & (grad_l * grad_h < 0.0)
        )
        _record_brackets(mask, float(lambda_edges[i]), float(lambda_edges[i + 1]), grad_l, grad_h)

    needs_fallback = bracket_count > 1

    # Solve interior roots for markers with exactly one bracket.
    single_idx = np.where(bracket_count == 1)[0]
    if single_idx.size:
        X_alt = np.empty((X_local.shape[0], X_local.shape[1] + 1), dtype=np.float64)
        X_alt[:, :-1] = X_local
        for marker_idx in single_idx:
            X_alt[:, -1] = G_local[:, marker_idx]
            state = _optimize_alt_h2_from_bracket(
                y_local,
                X_alt,
                eig_safe,
                lam_l=float(first_lam_l[marker_idx]),
                lam_h=float(first_lam_h[marker_idx]),
                grad_l=float(first_grad_l[marker_idx]),
                grad_h=float(first_grad_h[marker_idx]),
                tol=tol,
            )
            if (not state.success) or (not np.isfinite(state.neg_loglik)) or (state.beta_hat is None) or (state.x_vix is None):
                needs_fallback[marker_idx] = True
                continue
            best_valid[marker_idx] = True
            best_neg[marker_idx] = float(state.neg_loglik)
            best_h2[marker_idx] = float(state.h2)
            best_beta[marker_idx] = float(state.beta_hat[-1])
            best_y_pxy[marker_idx] = float(state.y_pxy)
            e_last = np.zeros(state.x_vix.shape[0], dtype=np.float64)
            e_last[-1] = 1.0
            inv_col_last = _solve_linear_system(state.x_vix, e_last)
            best_inv_b22[marker_idx] = float(inv_col_last[-1])

    # Emit results for markers resolved by shared states and/or bracket root solves.
    valid_best = best_valid & (~needs_fallback)
    if np.any(valid_best):
        lrt_stat = 2.0 * (float(null_neg_loglik) - best_neg[valid_best])
        lrt_stat = np.maximum(np.where(np.isfinite(lrt_stat), lrt_stat, 0.0), 0.0)
        p_values[valid_best] = stats.chi2.sf(lrt_stat, df=1)
        betas[valid_best] = best_beta[valid_best]
        df = max(1, n_samples - (X_local.shape[1] + 1))
        v_base = best_y_pxy[valid_best] / df
        var_marker = v_base * best_inv_b22[valid_best]
        se_vals = np.sqrt(np.maximum(var_marker, 0.0))
        finite = np.isfinite(se_vals) & np.isfinite(p_values[valid_best]) & np.isfinite(betas[valid_best])
        std_errors[valid_best] = np.where(finite, se_vals, np.inf)
        bad_emit_idx = np.where(valid_best)[0][~finite]
        if bad_emit_idx.size:
            needs_fallback[bad_emit_idx] = True

    # Conservative exact fallback for any unresolved or numerically unstable markers.
    fallback_idx = np.where(needs_fallback | (~best_valid))[0]
    if fallback_idx.size:
        X_alt = np.empty((X_local.shape[0], X_local.shape[1] + 1), dtype=np.float64)
        X_alt[:, :-1] = X_local
        for marker_idx in fallback_idx:
            X_alt[:, -1] = G_local[:, marker_idx]
            _, p_val, beta, se = fit_marker_lrt_prebuilt(
                y_local,
                X_alt,
                eig_safe,
                null_neg_loglik,
                null_h2=null_h2,
                solver_norm=solver_key,
                assume_sanitized=True,
            )
            p_values[marker_idx] = p_val
            betas[marker_idx] = beta
            std_errors[marker_idx] = se

    return p_values, betas, std_errors


def fit_marker_lrt_prebuilt(
    y_transformed: np.ndarray,
    X_alt: np.ndarray,
    eigenvals: np.ndarray,
    null_neg_loglik: float,
    *,
    null_h2: Optional[float] = None,
    solver: str = "GEMMA",
    assume_sanitized: bool = False,
    solver_norm: Optional[str] = None,
) -> Tuple[float, float, float, float]:
    """LRT with a prebuilt alternative design matrix `[X | g]`.

    When `assume_sanitized=True`, this skips per-call sanitization and is intended
    for trusted internal batch loops that already provide finite float arrays.
    """
    if assume_sanitized:
        y_local = y_transformed
        X_alt_local = X_alt
    else:
        y_local = _sanitize_array(y_transformed)
        X_alt_local = _sanitize_array(X_alt)

    solver_key = solver_norm if solver_norm is not None else _normalize_lrt_solver(solver)
    if solver_key == "AUTO":
        # Resolve once per marker based on sample size, then run the standard core
        # path without additional branching inside the inner solver routine.
        solver_key = "BRENT" if int(y_local.size) < 250 else "GEMMA"
    return _fit_marker_lrt_core(
        y_local,
        X_alt_local,
        eigenvals,
        null_neg_loglik,
        null_h2=null_h2,
        solver_norm=solver_key,
    )


def fit_marker_lrt(
    y_transformed: np.ndarray,
    X_transformed: np.ndarray,
    g_transformed: np.ndarray,
    eigenvals: np.ndarray,
    null_neg_loglik: float,
    *,
    null_h2: Optional[float] = None,
    solver: str = "GEMMA",
) -> Tuple[float, float, float, float]:
    """
    Perform Likelihood Ratio Test (LRT) for a single marker.
    
    Args:
        y_transformed: Phenotype vector in eigenspace (U'y)
        X_transformed: Covariate matrix in eigenspace (U'X) (Fixed effects)
        g_transformed: Genotype vector in eigenspace (U'g) (Marker effect)
        eigenvals: Eigenvalues of kinship matrix
        null_neg_loglik: Negative log-likelihood of the NULL model (pre-calculated)
        
    Returns:
        Tuple (LRT_statistic, p_value, beta_hat, se_hat)
    """
    
    # Construct Alternative Model Design Matrix: [X | g].
    if g_transformed.ndim == 1:
        g_col = g_transformed[:, np.newaxis]
    else:
        g_col = g_transformed

    X_alt = np.hstack([X_transformed, g_col])
    return fit_marker_lrt_prebuilt(
        _sanitize_array(y_transformed),
        _sanitize_array(X_alt),
        eigenvals,
        null_neg_loglik,
        null_h2=null_h2,
        solver=solver,
        assume_sanitized=True,
    )
