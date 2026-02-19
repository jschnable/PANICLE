"""Configuration for BAYESLOCO association testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class BayesLocoConfig:
    # task
    task: str = "quantitative"

    # marker filtering / M_effective definition
    maf_min: float = 0.0
    marker_missing_max: float = 1.0
    drop_monomorphic: bool = True

    # prior / model (mandatory h2-scaled empirical tuning)
    sigma_spike2: float = 1e-6
    h2_estimator: str = "he_mom"  # he_mom
    h2_min: float = 1e-4
    h2_max: float = 0.95
    h2_pair_sample: int = 20_000
    h2_marker_sample: int = 2048
    prior_tune_pi_grid: Tuple[float, ...] = (0.005, 0.02, 0.05)
    prior_tune_slab_scale_grid: Tuple[float, ...] = (0.75, 1.25)
    prior_tune_val_fraction: float = 0.1
    prior_tune_metric: str = "val_nll"  # val_nll|val_mse
    prior_tune_two_stage: bool = True
    prior_tune_stage1_marker_fraction: float = 0.2
    prior_tune_stage1_max_iter: int = 12
    prior_tune_stage1_patience: int = 3
    prior_tune_top_k: int = 3
    prior_tune_stage2_max_iter: int = 20
    prior_tune_stage2_patience: int = 4
    prior_tune_prune_after_epochs: int = 8
    prior_tune_prune_rel_gap: float = 0.01
    prior_tune_warm_start: bool = True
    prior_tune_max_iter: int = 40
    prior_tune_patience: int = 5

    # variance handling
    estimate_sigma_e2: bool = True
    sigma_e2_min: float = 1e-6

    # inference engine (v1 supports CAVI only)
    inference_engine: str = "cavi"  # cavi

    # convergence / objective schedule
    kl_anneal: bool = False
    kl_anneal_epochs: int = 20
    max_iter: int = 120
    tol_elbo: float = 1e-4
    patience: int = 8
    elbo_eval_interval: int = 2
    elbo_eval_interval_screened: int = 5

    # batching
    batch_markers_fit: int = 2048
    batch_markers_test: int = 8192
    batch_samples: Optional[int] = None

    # block-CAVI controls
    cavi_damping: float = 1.0
    loco_refine_iter: int = 40
    refine_patience: int = 8

    # active-set screening (default-on)
    active_set_screening: bool = True
    screening_warmup_epochs: int = 6
    screening_threshold: float = 3e-3
    verification_interval: int = 8
    screening_keep_top_k: int = 5000

    # optional marginal initialization
    marginal_init: bool = True
    marginal_init_p_threshold: float = 0.5

    # reserved controls for potential future SVI backend
    svi_learning_rate: float = 5e-2
    svi_min_learning_rate: float = 1e-4
    svi_gradient_clip: float = 5.0

    # LOCO behavior
    loco_mode: str = "subtract_only"  # subtract_only|refine
    freeze_non_loco_params: bool = True

    # testing / calibration
    test_method: str = "score"  # score|wald
    robust_se: bool = True
    residual_var_correction: str = "diag"  # none|diag
    calibrate_stat_scale: str = "none"  # none|gc|unrelated_subset
    unrelated_subset_indices: Optional[List[int]] = None
    unrelated_subset_min_n: int = 10000

    # execution
    backend: str = "numpy"  # future: cupy/torch
    dtype_compute: str = "float32"
    dtype_accum: str = "float64"
    random_seed: int = 42
    deterministic: bool = True

    # numeric safety
    eps: float = 1e-12

    @classmethod
    def from_object(cls, value: Optional["BayesLocoConfig | Dict[str, Any]"]) -> "BayesLocoConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise ValueError("bl_config must be None, a dict, or BayesLocoConfig")

    def validate(self) -> None:
        if self.task != "quantitative":
            raise ValueError("BAYESLOCO currently supports task='quantitative' only")
        if self.backend != "numpy":
            raise ValueError("BAYESLOCO v1 supports backend='numpy' only")
        if self.inference_engine != "cavi":
            raise ValueError("BAYESLOCO v1 supports inference_engine='cavi' only")
        if self.test_method not in {"score", "wald"}:
            raise ValueError("test_method must be 'score' or 'wald'")
        if self.loco_mode not in {"subtract_only", "refine"}:
            raise ValueError("loco_mode must be 'subtract_only' or 'refine'")
        if self.residual_var_correction not in {"none", "diag"}:
            raise ValueError("residual_var_correction must be 'none' or 'diag'")
        if self.calibrate_stat_scale not in {"none", "gc", "unrelated_subset"}:
            raise ValueError("calibrate_stat_scale must be 'none', 'gc', or 'unrelated_subset'")
        if self.prior_tune_metric not in {"val_nll", "val_mse"}:
            raise ValueError("prior_tune_metric must be 'val_nll' or 'val_mse'")
        if self.h2_estimator != "he_mom":
            raise ValueError("h2_estimator must be 'he_mom' in v1")

        if not (0.0 < self.h2_min < self.h2_max < 1.0):
            raise ValueError("Require 0 < h2_min < h2_max < 1")
        if not (0.0 < self.prior_tune_val_fraction < 0.5):
            raise ValueError("Require 0 < prior_tune_val_fraction < 0.5")
        if not (0.0 < self.prior_tune_stage1_marker_fraction <= 1.0):
            raise ValueError("Require 0 < prior_tune_stage1_marker_fraction <= 1")
        if self.prior_tune_prune_rel_gap < 0.0 or self.prior_tune_prune_rel_gap >= 1.0:
            raise ValueError("Require 0 <= prior_tune_prune_rel_gap < 1")
        if self.elbo_eval_interval <= 0 or self.elbo_eval_interval_screened <= 0:
            raise ValueError("elbo_eval_interval and elbo_eval_interval_screened must be positive")
        if self.screening_keep_top_k < 0:
            raise ValueError("screening_keep_top_k must be non-negative")

        if len(self.prior_tune_pi_grid) == 0 or any(x <= 0.0 or x >= 1.0 for x in self.prior_tune_pi_grid):
            raise ValueError("prior_tune_pi_grid entries must satisfy 0 < pi < 1")
        if len(self.prior_tune_slab_scale_grid) == 0 or any(x <= 0.0 for x in self.prior_tune_slab_scale_grid):
            raise ValueError("prior_tune_slab_scale_grid entries must be > 0")

        n_candidates = len(self.prior_tune_pi_grid) * len(self.prior_tune_slab_scale_grid)
        if self.prior_tune_top_k < 1:
            raise ValueError("prior_tune_top_k must be >= 1")
        if self.prior_tune_top_k > n_candidates:
            # Keep config valid when users provide a tiny candidate grid.
            self.prior_tune_top_k = n_candidates

        if self.batch_markers_fit <= 0 or self.batch_markers_test <= 0:
            raise ValueError("batch sizes must be positive")
        if self.max_iter <= 0 or self.patience <= 0:
            raise ValueError("max_iter and patience must be positive")
        if self.calibrate_stat_scale == "unrelated_subset":
            if self.unrelated_subset_indices is None:
                raise ValueError("unrelated_subset calibration requires unrelated_subset_indices")
            if len(self.unrelated_subset_indices) < self.unrelated_subset_min_n:
                raise ValueError(
                    "unrelated_subset_indices shorter than unrelated_subset_min_n "
                    f"({len(self.unrelated_subset_indices)} < {self.unrelated_subset_min_n})"
                )
