# BAYESLOCO GWAS Method Spec (Spike-and-Slab, CPU First, GPU-Ready)

## 1. Goal
Introduce a new GWAS method in PANICLE that models noninfinitesimal architecture using spike-and-slab Bayesian regression plus LOCO-aware testing, with mandatory data-driven `h2`-scaled prior tuning.

Requirements:
1. Available as a first-class GWAS method alongside `GLM`, `MLM`, `FARMCPU`, and `BLINK`.
2. CPU-only in v1.
3. Designed so GPU acceleration is an additive backend implementation (no major algorithm/module reorganization).

## 2. Scope
### In scope (v1)
1. Quantitative traits.
2. Spike-and-slab mean-field variational fit on CPU using block-CAVI as the default inference engine.
3. LOCO testing with score/Wald options.
4. Pipeline/core API integration and standard PANICLE outputs.
5. Deterministic mode with fixed seed and fixed batch order.
6. Mandatory data-driven tuning of prior sparsity (`pi`) and slab scale under `h2`-scaled priors.

### Out of scope (v1)
1. Binary-trait logistic model, SPA, Firth fallback.
2. GPU execution.
3. Distributed multi-node execution.
4. Fully Bayesian hyperpriors over `pi`/slab-scale (v1 uses empirical data-driven tuning, then fixes selected values).
5. Natural-gradient methods and fully asynchronous coordinate updates (planned for later versions).

## 3. Public API
### 3.1 New association entry point
Add:

```python
PANICLE_BayesLOCO(
    phe,
    geno,
    map_data,
    CV=None,
    cpu=1,
    verbose=True,
    bl_config=None,
)
```

Return type: `AssociationResults`.

Notes:
1. `phe`, `geno`, `CV`, and `map_data` follow existing `PANICLE_MLM_LOCO` conventions.
2. `loco_kinship` is intentionally not part of this API (BayesLOCO is not kinship-driven).
3. `maxLine` is intentionally not part of this API; marker batching is controlled by `bl_config.batch_markers_fit` and `bl_config.batch_markers_test`.

### 3.2 Method name in orchestration
1. Canonical method string: `BAYESLOCO`.
2. No aliases in v1.
3. Report/display label: `BayesLOCO`.

Dispatch updates required in:
1. `panicle/pipelines/gwas.py`
2. `panicle/core/mvp.py`
3. `panicle/association/__init__.py`
4. `panicle/__init__.py`

### 3.3 Behavior constraints
1. Missing `map_data` is an error in v1 (LOCO required).
2. Non-finite phenotype/covariate handling matches existing MLM/LOCO validation patterns.
3. Binary traits raise `NotImplementedError` with explicit guidance.
4. BayesLOCO must not trigger kinship computation paths in pipeline/core (`need_kinship` remains MLM-only).

### 3.4 Result object and metadata
Base return remains `AssociationResults` (`effects`, `se`, `pvalues`), with a required metadata extension for BayesLOCO diagnostics.

Required metadata keys:
1. `method="BAYESLOCO"`
2. `loco_mode`
3. `calibration_mode`
4. `lambda_gc_raw`, `lambda_gc_final`
5. `elbo_trace`
6. `converged`
7. `sigma_e2_final`
8. `n_markers_fit`
9. `h2_hat`
10. `prior_pi_selected`
11. `prior_slab_scale_selected`
12. `sigma_slab2_selected`
13. `prior_tuning_metric`, `prior_tuning_score`
14. `timing_total_s`, `timing_prior_tune_s`, `timing_main_fit_s`, `timing_loco_test_s`
15. `pass_equiv_total`, `pass_equiv_prior_tune`, `pass_equiv_main_fit`, `pass_equiv_loco_refine`
16. `active_markers_trace`
17. `prior_tune_candidates_stage1`, `prior_tune_candidates_stage2`

Implementation options:
1. Add optional `metadata: Dict[str, Any]` to `AssociationResults`, or
2. Add a `BayesLocoResults` wrapper with `.association_results` plus metadata.

## 4. Statistical method (v1)
### 4.1 Working space and notation (FWL residualized space)
All fitting and testing are performed in covariate-residualized space using Frisch-Waugh-Lovell (FWL), not raw-phenotype space.

Definitions:
1. Covariate matrix: `X` (includes intercept).
2. Thin/economic QR decomposition: `X = Q R`, with `Q` shape `(n, p)` (`np.linalg.qr(X, mode='reduced')` equivalent).
3. Projection operator (implicit): `M_X v = v - Q(Q'v)`.
4. Residualized phenotype: `r = M_X y`.
5. Residualized marker vectors: `z_j = M_X g_j`.

Scale conventions:
1. Fitting uses residualized standardized markers `z_j_std` (for optimizer stability).
2. LOCO prediction (`yhat_total`, `yhat_chr`, `yhat_not_c`) and uncertainty terms (`v_uncert_c`) are computed in standardized-marker space (`z_j_std`) to stay unit-consistent with `m_j` and `v_j`.
3. Final per-marker testing uses residualized unstandardized markers `z_j` to produce per-allele effects consistent with existing PANICLE methods.
4. If an implementation uses standardized markers during final testing for speed, it must back-transform effects and SE to per-allele scale before populating `AssociationResults`.

Important:
1. Do not materialize `M_X` as `n x n`.
2. Once in FWL space, no second covariate-projection step should be applied.

### 4.2 Prior, variational family, and effective coefficient moments
Spike-and-slab approximation:
1. Inclusion variable: `h_j ~ Bernoulli(pi)`, where `pi` is selected by mandatory data-driven prior tuning.
2. `beta_j | h_j=1 ~ N(0, sigma_slab2)`.
3. `beta_j | h_j=0 ~ N(0, sigma_spike2)`, with `sigma_spike2 << sigma_slab2`.

Variational family (mean-field):
1. `q(h_j) = Bernoulli(phi_j)`.
2. `q(beta_j) = N(mu_j, s_j^2)`.

Define effective coefficient `b_j = h_j * beta_j`:
1. `m_j = E_q[b_j] = phi_j * mu_j`.
2. `v_j = Var_q(b_j) = phi_j * (s_j^2 + mu_j^2) - (phi_j * mu_j)^2`.

Mandatory `h2`-scaled prior mode (authoritative):
1. Estimate `h2_hat` before main fit using residualized phenotype `r` and standardized markers `Z_std` (v1 estimator: HE-style method-of-moments).
2. Clip `h2_hat` into `[h2_min, h2_max]`.
3. Let `var_r = Var(r)` on the tuning-train subset (or full training set when no split is used).
4. `M_effective = number of markers retained for BayesLOCO fitting after BayesLOCO marker QC/filtering`.
5. For candidate `(pi, kappa)`:
   `sigma_slab2(pi, kappa) = kappa * (h2_hat * var_r) / max(pi * M_effective, 1.0)`.
6. `sigma_spike2` remains fixed and small (continuous spike approximation).

Mandatory data-driven prior tuning:
1. Candidate grids:
   - `pi in prior_tune_pi_grid`
   - `kappa in prior_tune_slab_scale_grid`
2. Build deterministic train/validation split in FWL space using `prior_tune_val_fraction` (seeded by `random_seed`).
3. Two-stage tuning is default (`prior_tune_two_stage=True`):
   - Stage 1 (coarse): evaluate all candidates on deterministic marker subset (`prior_tune_stage1_marker_fraction`) using `prior_tune_stage1_max_iter` and `prior_tune_stage1_patience`.
   - Stage 1 pruning: after `prior_tune_prune_after_epochs`, drop candidates worse than current best by relative gap `prior_tune_prune_rel_gap`.
   - Stage 2 (full): evaluate top `prior_tune_top_k` candidates on full marker set with `prior_tune_stage2_max_iter` and `prior_tune_stage2_patience`.
4. If `prior_tune_two_stage=False`, evaluate full grid with `prior_tune_max_iter` and `prior_tune_patience`.
5. Candidate ordering is deterministic (`pi` ascending, then `kappa` ascending).
6. Warm-start across candidate fits is required when `prior_tune_warm_start=True`.
7. Score candidates by `prior_tune_metric`:
   - `val_nll` (default): Gaussian validation NLL in FWL space.
   - `val_mse`: validation mean squared error of `r_val - Z_val_std * m`.
8. Select `(pi*, kappa*)` minimizing validation metric (deterministic tie-break: smaller `pi`, then smaller `kappa`).
9. Main fit and all LOCO folds use `pi*` and `sigma_slab2* = sigma_slab2(pi*, kappa*)`.

### 4.3 ELBO objective (normative)
Optimization target:
1. `ELBO = E_q[log p(r | b, sigma_e2)] - tau * KL_total`, where `tau` is KL weight.
2. If `kl_anneal=False`, `tau=1.0` for all epochs.
3. If `kl_anneal=True`, `tau = min(1.0, (epoch + 1) / kl_anneal_epochs)`.
4. Under mean-field with `m_j`/`v_j` moments (Section 4.2), likelihood term should match:
   `E_q[log p(r|b,sigma_e2)] = -n/2*log(2*pi_const*sigma_e2) - ( ||r - Z_std*m||^2 + sum_j v_j*||z_j_std||^2 )/(2*sigma_e2)`,
   up to algebraically equivalent constants.
   Here `pi_const` is the mathematical constant `3.14159...`.
5. With serial CAVI updates, ELBO should be non-decreasing up to floating-point noise.
6. With block-CAVI (stale residual within a block), small local ELBO dips are acceptable, but epoch-level trend should be non-decreasing after warm-up.

KL decomposition for v1 mean-field spike-and-slab:
1. `KL_total = sum_j KL_Bernoulli(phi_j || pi_selected)`
2. `+ sum_j phi_j * KL_Normal(N(mu_j, s_j^2) || N(0, sigma_slab2_selected))`
3. `+ sum_j (1 - phi_j) * KL_Normal(N(mu_j, s_j^2) || N(0, sigma_spike2))`

Implementations may use equivalent algebraic forms, but must be numerically equivalent.

### 4.4 Residual variance update (`sigma_e2`)
`sigma_e2` must be estimated with a variationally consistent update:

`sigma_e2 <- max(sigma_e2_min, ( ||r - Z_std * m||^2 + sum_j( v_j * ||z_j_std||^2 ) ) / n )`

where:
1. `m` is vector of `m_j = phi_j * mu_j`.
2. `v_j` is defined in Section 4.2.

Streaming requirement:
1. Do not run a second full genotype pass only for `sigma_e2`.
2. Accumulate residual and variance terms online during the epoch's streaming/block pass (Welford-style or equivalent numerically stable accumulation).

### 4.5 Fitting algorithm
Default inference engine is block-CAVI (`inference_engine="cavi"`).

Core loop:
1. Build `r = M_X y` once.
2. Stream genotype blocks, impute missing values as needed, and standardize once (cached stats).
3. Run mandatory prior tuning (Section 4.2) to select `pi*` and `sigma_slab2*` using `h2`-scaled candidates.
4. Initialize main-fit state (optionally from best tuning candidate checkpoint).
5. Maintain running residual `r_resid = r - Z_std * m`.
6. For each marker block `B`, compute cavity residuals and update marker variational parameters with closed-form coordinate updates.
7. Optional damping may be applied before residual correction:
   `m_B_new <- cavi_damping * m_B_new + (1 - cavi_damping) * m_B_old`.
8. Update block moments and apply residual correction:
   `r_resid <- r_resid - Z_B_std * (m_B_new - m_B_old)`.
9. Update `sigma_e2` at epoch end (Section 4.4) and compute ELBO on cadence:
   - every `elbo_eval_interval` epochs before screening stability,
   - every `elbo_eval_interval_screened` epochs in stable screened mode.
10. Stop by `tol_elbo`, `patience`, and `max_iter`.

Closed-form update requirement:
1. Implementers must use closed-form CAVI updates for the chosen mean-field parameterization (no gradient-based optimizer in default engine).
2. Residual-dependent updates should use precomputed marker norms/crossproducts for block efficiency.
3. Reference per-marker update pattern:
   - `r_j = r_resid + z_j_std * m_j_old` (cavity residual)
   - `u_j = z_j_std' r_j`
   - `d_j_std = ||z_j_std||^2`
   - `s_j^2 = 1 / (d_j_std / sigma_e2 + 1 / sigma_prior_j)` (parameterization-specific prior precision)
   - `mu_j = s_j^2 * (u_j / sigma_e2)`
   - `phi_j` update uses closed-form spike/slab log-odds under the chosen approximation
   - `m_j_new = phi_j * mu_j`
   - `r_resid <- r_resid - z_j_std * (m_j_new - m_j_old)`
4. In block-CAVI, updates may use stale residuals within block and apply residual correction once per block.

Active-set screening (default-on):
1. Warm-up phase: full-marker updates for `screening_warmup_epochs`.
2. Screening phase: markers with `phi_j < screening_threshold` are frozen, but top `screening_keep_top_k` markers by `phi_j` remain active.
3. Verification phase: every `verification_interval` screened epochs, run a full sweep to allow reactivation.
4. Screened mode must preserve correctness by updating residuals with all active markers and any reactivated markers.

Optional marginal initialization (recommended for large `M`):
1. Run one marginal pass to initialize `phi_j`/`m_j` before CAVI epochs (`marginal_init=True`).
2. Initialization may use FWL marginal statistics and optional `marginal_init_p_threshold` for early freezing candidates.

Experimental fallback:
1. `inference_engine="svi_experimental"` may use Adam-SVI with separate `svi_*` config fields.
2. SVI is non-default and not the primary optimization path for v1 performance claims.

Determinism when `deterministic=True`:
1. Marker/block permutation is seeded and fixed.
2. Reduction ordering is fixed.
3. Thread/BLAS nondeterminism controls are applied/documented.

### 4.6 LOCO prediction modes
1. `loco_mode="subtract_only"` (default):
   `yhat_not_c = yhat_total - yhat_chr[c]`, where
   `yhat_total = sum_j z_j_std * m_j` and
   `yhat_chr[c] = sum_{j in c} z_j_std * m_j`.
   Implementations should accumulate `yhat_total` and per-chromosome `yhat_chr[c]` during streaming fit passes to avoid extra genotype I/O.
   Equivalent unstandardized form is allowed only if coefficients are back-transformed first: `beta_per_allele_j = m_j / scale_j`.

2. `loco_mode="refine"`:
   Whole-genome warm-start, then short LOCO refinement.
   v1 default in refine mode: `freeze_non_loco_params=True`.
   Refinement runs up to `loco_refine_iter` with early stopping enabled (`tol_elbo`, `refine_patience`).

Both modes must be benchmarked; v1 default is `subtract_only`.

### 4.7 Testing step (score and Wald) and `sigma_test2`
For chromosome `c`, define LOCO residual phenotype in FWL space:
1. `r_c = r - yhat_not_c`.

Use residualized unstandardized chromosome matrix `Z_c`.

Precompute denominator terms once per marker:
1. `d_j = z_j' z_j`.
2. Efficient QR form: `d_j = ||g_j||^2 - ||Q' g_j||^2`.

Compute block numerators via BLAS:
1. `u_block = Z_c' r_c`.

Define per-chromosome test variance:
1. Base variance:
   - `sigma_base2_c = sigma_e2` for `loco_mode="subtract_only"`.
   - `sigma_base2_c = sigma_e2_c` for `loco_mode="refine"` when fold-specific `sigma_e2` is updated; otherwise use global `sigma_e2`.
2. Uncertainty correction term (`residual_var_correction="diag"`):
   `v_uncert_c = (1/n) * sum_{j not in c} v_j * ||z_j_std||^2`.
   Streaming-efficient computation should use LOCO subtraction:
   - `V_total = sum_all v_j * ||z_j_std||^2`
   - `V_chr[c] = sum_{j in c} v_j * ||z_j_std||^2`
   - `v_uncert_c = (V_total - V_chr[c]) / n`
   (Equivalent unstandardized form: `(1/n) * sum_{j not in c} v_j * ||z_j||^2 / scale_j^2`.)
3. Final variance:
   - If correction `none`: `sigma_test2_c = sigma_base2_c`.
   - If correction `diag`: `sigma_test2_c = sigma_base2_c * (1 + v_uncert_c / max(sigma_base2_c, eps))`.

Score test (`test_method="score"`):
1. `T_j = u_j^2 / (sigma_test2_c * d_j)`.
2. p-value from `chi2(df=1)`.

Wald test (`test_method="wald"`):
1. `beta_hat_j = u_j / d_j`.
2. Classical SE: `se_j = sqrt(sigma_test2_c / d_j)`.
3. `t_j = beta_hat_j / se_j`, two-sided p-value from chosen approximation (must be documented).

For homoskedastic linear model in FWL space, score and Wald are asymptotically equivalent; finite-sample differences must be documented.

### 4.8 Robust SE and calibration
`robust_se` (v1):
1. Wald path: HC1 sandwich variance in FWL space.
2. Score path: sandwich score variance when implemented; otherwise document explicit fallback.

Statistical scale calibration (`calibrate_stat_scale`):
1. `none`: no post-hoc scaling.
2. `gc` (default): genomic-control scaling.
3. `unrelated_subset`: scale from unrelated homogeneous subset.

Data pathway for unrelated subset:
1. `bl_config.unrelated_subset_indices` (global sample indices after matching/filtering).
2. If missing/too small (`< unrelated_subset_min_n`), fallback to `gc` with warning.

### 4.9 Computational cost model and accounting (normative)
Define marker-pass-equivalent accounting:
1. Let `M_effective` be number of fit markers after BayesLOCO QC.
2. For any epoch `e`, define touched markers `M_touched_e` (including active-set and verification behavior).
3. `pass_equiv_e = M_touched_e / max(M_effective, 1)`.
4. `pass_equiv_main_fit = sum_e pass_equiv_e` over main-fit epochs.
5. `pass_equiv_prior_tune = sum_{candidate,epoch} pass_equiv_{candidate,epoch}` over tuning runs.
6. `pass_equiv_loco_refine = sum_{chrom,epoch} pass_equiv_{chrom,epoch}` for `loco_mode="refine"`; zero for `subtract_only`.
7. `pass_equiv_total = pass_equiv_prior_tune + pass_equiv_main_fit + pass_equiv_loco_refine`.

Runtime decomposition that must be reported:
1. `timing_prior_tune_s`
2. `timing_main_fit_s`
3. `timing_loco_test_s`
4. `timing_total_s` (sum of above plus overhead)

Operational objective:
1. Default settings should target low `pass_equiv_total` first (I/O-dominant regime), before micro-optimizing kernel flops.
2. Any performance claim must include both wall time and pass-equivalent breakdown.

### 4.10 Known limitations (v1)
1. Uncertainty propagation with `diag` correction is approximate and may miscalibrate in extreme sparse large-effect regimes.

## 5. Architecture (GPU-ready by design)
### 5.1 Module layout (reduced v1 surface area)
Recommended v1 files:
1. `panicle/association/bayes_loco/__init__.py`
2. `panicle/association/bayes_loco/api.py`
3. `panicle/association/bayes_loco/config.py`
4. `panicle/association/bayes_loco/engine.py`          (fit + loco + test orchestration)
5. `panicle/association/bayes_loco/data.py`            (streaming, imputation, standardization, projections)
6. `panicle/association/bayes_loco/state.py`
7. `panicle/association/bayes_loco/diagnostics.py`
8. `panicle/association/bayes_loco/backends/base.py`
9. `panicle/association/bayes_loco/backends/numpy_backend.py`
10. `panicle/association/bayes_loco/backends/factory.py`
11. `panicle/association/bayes_loco.py`                (thin re-export shim)

Notes:
1. Additional splitting (`fit.py`, `loco.py`, `test_quant.py`) is optional after v1 stabilization.

### 5.2 Backend abstraction contract
`ArrayBackend` required ops:
1. Creation/conversion: `asarray`, `zeros`, `ones`, `empty`, `astype`.
2. Math: `matmul`, `dot`, `einsum`, `sum`, `mean`, `sqrt`, `exp`, `log`, `clip`, `where`.
3. Norm helpers: `sumsq` and/or `norm`.
4. Gather/scatter/index helpers for blocks.
5. RNG: seeded normal/uniform.
6. Utility: `to_host`, `is_gpu`, dtype metadata.

Rules:
1. Core algorithm code uses backend ops only.
2. v1 backend is `NumpyBackend`.
3. Future GPU backend plugs into factory with no core API changes.

### 5.3 Data flow, missing data, and memory invariants
`data.py` responsibilities:
1. Handle missing genotype values consistently with existing PANICLE behavior (`GenotypeMatrix` imputation path or `impute_numpy_batch_major_allele` for numpy blocks).
2. Define and apply BayesLOCO marker QC/filtering used to compute `M_effective`.
3. Cache marker means/scales once; reuse across LOCO folds.
4. Build FWL projectors via QR (`Q`, `R`) once; do not form `M_X` explicitly.
5. Precompute/cache per-marker denominators `d_j`.
6. Maintain streaming-first invariant: never materialize full standardized `G` by default.
7. Optional full-cache mode may be allowed only behind explicit flag with memory check.

### 5.4 Parallelism and precision strategy
Parallelism policy:
1. Prioritize chromosome-level parallelism in testing (embarrassingly parallel after fit).
2. Fit phase (block-CAVI) should avoid cross-block parallelism due residual dependencies; parallelism is allowed within block linear algebra kernels.
3. Never nest outer and inner parallelism.

Precision policy:
1. Variational updates and large matmuls in float32.
2. ELBO accumulators and final test statistic/p-value paths in float64.
3. Avoid repeated bulk casting of full arrays.

## 6. Configuration (authoritative defaults)
Defaults in this section are authoritative. Section 13 is presets only.

```python
from typing import Optional, List, Tuple

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
    h2_estimator: str = "he_mom"           # he_mom
    h2_min: float = 1e-4
    h2_max: float = 0.95
    prior_tune_pi_grid: Tuple[float, ...] = (0.005, 0.02, 0.05)
    prior_tune_slab_scale_grid: Tuple[float, ...] = (0.75, 1.25)
    prior_tune_val_fraction: float = 0.1
    prior_tune_metric: str = "val_nll"     # val_nll|val_mse
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

    # inference engine
    inference_engine: str = "cavi"        # cavi|svi_experimental

    # convergence / objective schedule
    kl_anneal: bool = False
    kl_anneal_epochs: int = 20            # tau = min(1, (epoch+1)/kl_anneal_epochs) when enabled
    max_iter: int = 120
    tol_elbo: float = 1e-4
    patience: int = 8
    elbo_eval_interval: int = 2
    elbo_eval_interval_screened: int = 5

    # batching
    batch_markers_fit: int = 4096
    batch_markers_test: int = 16384
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

    # optional SVI fallback controls (used only when inference_engine='svi_experimental')
    svi_learning_rate: float = 5e-2
    svi_min_learning_rate: float = 1e-4
    svi_gradient_clip: float = 5.0

    # LOCO behavior
    loco_mode: str = "subtract_only"      # subtract_only|refine
    freeze_non_loco_params: bool = True

    # testing / calibration
    test_method: str = "score"            # score|wald
    robust_se: bool = True
    residual_var_correction: str = "diag" # none|diag
    calibrate_stat_scale: str = "gc"      # none|gc|unrelated_subset
    unrelated_subset_indices: Optional[List[int]] = None
    unrelated_subset_min_n: int = 10000

    # execution
    backend: str = "numpy"                # future: cupy/torch
    dtype_compute: str = "float32"
    dtype_accum: str = "float64"
    random_seed: int = 42
    deterministic: bool = True
```

Validation requirements:
1. Strict bounds/enums for all config fields.
2. Reject unsupported backend in v1.
3. Reject `task != "quantitative"` in v1.
4. Convert `unrelated_subset_indices` to numpy integer array internally at runtime.
5. When `inference_engine='svi_experimental'`, use only `svi_*` optimizer fields and ignore CAVI-only controls with warning.
6. Validate `0 < h2_min < h2_max < 1`.
7. Validate positive `prior_tune_pi_grid` and `prior_tune_slab_scale_grid` entries.
8. Validate `0 < prior_tune_val_fraction < 0.5`.
9. Validate `0 < prior_tune_stage1_marker_fraction <= 1`.
10. Validate `1 <= prior_tune_top_k <= len(prior_tune_pi_grid) * len(prior_tune_slab_scale_grid)`.
11. Validate `0 <= prior_tune_prune_rel_gap < 1`.
12. Validate positive `elbo_eval_interval` and `elbo_eval_interval_screened`.
13. Validate non-negative `screening_keep_top_k`.

## 7. Detailed implementation plan
### 7.1 Phase 1: Scaffolding and API plumbing
1. Add BayesLOCO package files and config.
2. Register `BAYESLOCO` dispatch in pipeline/core.
3. Add exports and docs stubs.
4. Add unsupported-path errors (binary, missing map).
5. Ensure pipeline/core do not trigger kinship paths for BayesLOCO.

### 7.2 Phase 2: CPU backend and data pipeline
1. Implement `NumpyBackend` + factory.
2. Implement QR/FWL projection helpers.
3. Implement streaming block iterators, missing-data imputation, and one-time standardization cache.
4. Implement `M_effective` marker filter accounting and denominator cache.
5. Implement `h2` estimator in FWL standardized-marker space (`he_mom`).

### 7.3 Phase 3: Variational fit core
1. Implement state container and block-CAVI updates.
2. Implement mandatory prior tuner over `(pi, slab_scale)` with two-stage evaluation, early pruning, deterministic tie-breaking, and warm-start reuse.
3. Implement ELBO accumulation and variationally consistent `sigma_e2` update.
4. Implement active-set screening and verification sweeps.
5. Implement optional marginal initialization pass.
6. Implement deterministic batching.
7. Optional: implement KL warm-up (`kl_anneal`) and SVI experimental fallback path.
8. Implement pass-equivalent accounting counters for tuning and main fit.

### 7.4 Phase 4: LOCO prediction and testing
1. Implement `subtract_only` fast mode (required).
2. Implement optional `refine` mode.
3. Implement score and Wald test paths (including explicit `sigma_test2_c` logic and robust SE behavior).
4. Implement `residual_var_correction` and `calibrate_stat_scale` modes.
5. Implement pass-equivalent accounting for LOCO refinement mode.
6. Emit required results metadata (including timing/cost counters).

### 7.5 Phase 5: Performance and stability
1. Benchmark and tune fit/testing parallel heuristics.
2. Confirm no full standardized-matrix materialization in default mode.
3. Add optional Numba kernels where beneficial.
4. Benchmark `subtract_only` vs `refine` and publish tradeoffs.
5. Benchmark active-set screening speedup vs full-pass CAVI.
6. Benchmark prior-tuning overhead and warm-start reuse gains.
7. Track and optimize `pass_equiv_total` as the primary fit-cost KPI.

### 7.6 Phase 6: Validation and docs
1. Run calibration/power simulation suite.
2. Add integration tests in pipeline/core pathways.
3. Document known limitations and parameter guidance.

## 8. Testing strategy
### 8.1 Unit tests
1. Backend API contract (`einsum`, `sumsq`/`norm`, block math).
2. Config validation and enum handling.
3. Determinism tests (fixed seed + fixed batch order).
4. Missing-data handling parity with existing imputation helpers.
5. `sigma_e2` update correctness (includes posterior variance term using `m_j=phi_j*mu_j` and `v_j`).
6. Per-allele effect-scale check for testing path (or correct back-transform when standardized markers are used).
7. CAVI update sanity checks: residual-update consistency and finite parameter bounds.
8. Active-set screening reactivation correctness after verification sweeps.
9. Prior formula test: `sigma_slab2 = kappa * (h2_hat * var_r) / max(pi * M_effective, 1)`.
10. Prior tuner determinism test: fixed seed -> same `(pi*, kappa*)`.
11. Two-stage tuner test: stage-1 pruning and top-k routing are deterministic.
12. Cost accounting test: `pass_equiv_total = pass_equiv_prior_tune + pass_equiv_main_fit + pass_equiv_loco_refine`.

### 8.2 Statistical tests
1. Null simulations: FPR at multiple thresholds.
2. Polygenicity sweep: `pi_true` in `{0.001, 0.01, 0.1, 0.5}`.
3. Sparse architecture: expected power gain vs `MLM`.
4. Infinitesimal architecture: non-inferiority vs `MLM`.
5. Calibration ablation: `residual_var_correction=none` vs `diag`.
6. Mode ablation: `subtract_only` vs `refine`.
7. Score vs Wald parity check under homoskedastic simulated linear model.
8. CAVI vs SVI fallback parity on small synthetic datasets (same seed/settings, tolerance-bounded differences).
9. Prior tuning recovery: selected `pi*` tracks simulated sparsity trend; selected slab scale stays stable under phenotype rescaling.

### 8.3 Integration tests
1. `GWASPipeline.run_analysis(methods=['BAYESLOCO'])`.
2. Mixed runs with `GLM/MLM/FARMCPU/BLINK/BAYESLOCO`.
3. Summary/output generation includes BayesLOCO and metadata.

### 8.4 Performance regression tests
Track:
1. Runtime.
2. Peak RSS.
3. Deterministic output drift bounds.
4. Runtime split: fit vs testing; `subtract_only` vs `refine`.
5. Full-pass epochs vs screened epochs and effective-marker throughput.
6. Pass-equivalent breakdown: tuning vs main-fit vs LOCO refinement.

## 9. Acceptance criteria (v1)
1. Public API + orchestration integration complete.
2. Quantitative analyses pass on existing example datasets.
3. Null calibration controlled in internal sims.
4. Demonstrated power gain in at least one sparse setting.
5. Runtime/memory baselines documented.
6. CPU-only install and execution.
7. BayesLOCO metadata emitted and validated.
8. Timing and pass-equivalent cost metadata emitted and validated.

## 10. GPU-ready guarantees
Hard requirements:
1. Core algorithm modules do not call `numpy` directly.
2. Backend factory is the only backend switch.
3. Core kernels consume/return backend arrays.
4. Host conversion is isolated to API/reporting boundaries.
5. Streaming-first default is backend-agnostic.

## 11. Risks and mitigations
1. **Convergence instability:** CAVI damping/early stopping; SVI-only controls (`svi_gradient_clip`, LR floor) when fallback engine is used.
2. **Prior-selection sensitivity:** mandatory data-driven tuning, deterministic split/tie-break rules, metadata diagnostics.
3. **Calibration risk:** uncertainty correction + calibration modes + simulation gates.
4. **Runtime growth:** block-CAVI default, active-set screening, streaming caches, subtract-only default.

## 12. Documentation updates
1. `README.md`: add BayesLOCO method overview and example.
2. `docs/api_reference.md`: add `PANICLE_BayesLOCO` and config table.
3. `docs/output_files.md`: add BayesLOCO outputs + metadata fields.
4. Changelog: mark as experimental in first release.

## 13. Optional presets (non-authoritative)
Section 6 defaults remain authoritative.

1. `preset="default"`:
   Uses Section 6 defaults.

2. `preset="throughput"`:
   `batch_markers_fit=8192`, `batch_markers_test=32768`, `max_iter=90`, `tol_elbo=3e-4`, `patience=6`, `prior_tune_stage1_max_iter=8`, `prior_tune_stage2_max_iter=12`, `prior_tune_top_k=2`, `loco_mode="subtract_only"`, `screening_threshold=5e-3`.

3. `preset="refine"`:
   `loco_mode="refine"`, `loco_refine_iter=40`, `freeze_non_loco_params=True`.

## 14. Backward compatibility
1. Existing methods/outputs remain unchanged.
2. No default method-list change in v1.
3. `BAYESLOCO` is opt-in until calibration/power/perf gates are met.
