
import pytest
import numpy as np
from scipy import stats
from panicle.association.glm import PANICLE_GLM
import pandas as pd
from panicle.association.mlm_loco import PANICLE_MLM_LOCO

def test_glm_statistical_correctness():
    """Verify GLM p-values against standard scipy.stats regression"""
    rng = np.random.default_rng(42)
    n = 100
    n_markers = 10

    # Random phenotype
    y = rng.standard_normal(n)
    phe = np.column_stack([np.arange(n), y])

    # 1. Test Simple Regression (No covariates)
    geno = rng.integers(0, 3, size=(n, n_markers)).astype(float)

    # Run MVP GLM
    res = PANICLE_GLM(phe, geno, cpu=1, verbose=False)

    # Check each marker manually
    for j in range(n_markers):
        g = geno[:, j]
        slope, intercept, r_value, p_value_t, std_err = stats.linregress(g, y)

        # Note: MVP GLM uses normal approximation for speed instead of t-distribution.
        # Compute normal-based p-value from t-statistic for comparison.
        t_stat = slope / std_err
        p_value = 2 * stats.norm.sf(np.abs(t_stat))

        np.testing.assert_allclose(res.pvalues[j], p_value, rtol=1e-5,
            err_msg=f"P-value mismatch at marker {j}")

def test_glm_covariates_correctness():
    """Verify GLM with covariates (Partial Regression)"""
    rng = np.random.default_rng(1337)
    n = 200
    n_markers = 5
    n_cov = 2

    # Co-variates
    CV = rng.standard_normal((n, n_cov))

    # Phenotype with covariate effects
    y = CV @ np.array([0.5, -0.5]) + rng.standard_normal(n)
    phe = np.column_stack([np.arange(n), y])

    geno = rng.integers(0, 3, size=(n, n_markers))

    res = PANICLE_GLM(phe, geno, CV=CV, verbose=False)

    # Validation using statsmodels (OLS)
    import statsmodels.api as sm

    for j in range(n_markers):
        g = geno[:, j]
        # Design matrix: Intercept + CV + g
        X = np.column_stack([np.ones(n), CV, g])
        model = sm.OLS(y, X).fit()

        # The last parameter is 'g'
        # Note: MVP GLM uses normal approximation for speed instead of t-distribution.
        # Compute normal-based p-value from t-statistic for comparison.
        t_val = model.tvalues[-1]
        p_val_norm = 2 * stats.norm.sf(np.abs(t_val))

        np.testing.assert_allclose(res.pvalues[j], p_val_norm, rtol=1e-5,
             err_msg=f"Covariate P-value mismatch at marker {j}")

def test_mlm_returns_reasonable_values():
    """Smoke test for MLM (statistical correctness is harder to unit-test simply)"""
    rng = np.random.default_rng(99)
    n = 50
    m = 100
    geno = rng.integers(0, 3, size=(n, m))
    y = rng.standard_normal(n)
    phe = np.column_stack([np.arange(n), y])
    
    map_df = pd.DataFrame({
        'SNP': [f'SNP{i:04d}' for i in range(m)],
        'CHROM': [f'Chr{(i % 5) + 1:02d}' for i in range(m)],
        'POS': np.arange(m) + 1
    })

    res = PANICLE_MLM_LOCO(phe, geno, map_data=map_df, verbose=False)
    
    assert len(res.pvalues) == m
    assert np.all(res.pvalues >= 0) and np.all(res.pvalues <= 1)
    # Check nothing is NaN (unless variance is 0)
    assert not np.any(np.isnan(res.pvalues))
