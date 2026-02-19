"""State containers for BAYESLOCO fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class CaviState:
    """Variational state over fitted markers."""

    m: np.ndarray
    v: np.ndarray
    phi: np.ndarray
    mu_slab: np.ndarray
    s2_slab: np.ndarray
    mu_spike: np.ndarray
    s2_spike: np.ndarray
    sigma_e2: float
    elbo_trace: List[float] = field(default_factory=list)
    active_markers_trace: List[int] = field(default_factory=list)
    converged: bool = False
    n_epochs: int = 0


@dataclass
class FitCost:
    """Runtime/cost accounting."""

    timing_prior_tune_s: float = 0.0
    timing_main_fit_s: float = 0.0
    timing_loco_test_s: float = 0.0
    timing_total_s: float = 0.0

    pass_equiv_prior_tune: float = 0.0
    pass_equiv_main_fit: float = 0.0
    pass_equiv_loco_refine: float = 0.0
    pass_equiv_total: float = 0.0
