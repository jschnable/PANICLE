import numpy as np
import pytest
from scipy import stats

from panicle.utils import stats as stats_utils


def test_bonferroni_correction_scales_threshold() -> None:
    pvalues = np.array([0.01, 0.5, 0.02])

    corrected, threshold = stats_utils.bonferroni_correction(pvalues, alpha=0.05)

    assert threshold == pytest.approx(0.05 / 3)
    np.testing.assert_allclose(corrected, np.array([0.03, 1.0, 0.06]))


def test_fdr_correction_bh_procedure() -> None:
    pvalues = np.array([0.001, 0.01, 0.2, 0.5])

    rejected, corrected = stats_utils.fdr_correction(pvalues, alpha=0.05)

    np.testing.assert_array_equal(rejected, np.array([True, True, False, False]))
    np.testing.assert_allclose(
        corrected,
        np.array([0.004, 0.02, 0.26666667, 0.5]),
    )


def test_calculate_maf_from_genotypes_handles_missing() -> None:
    genotypes = np.array(
        [
            [0, 1, -9],
            [2, -9, -9],
            [0, 1, 2],
        ],
        dtype=float,
    )

    maf = stats_utils.calculate_maf_from_genotypes(genotypes, missing_value=-9, max_dosage=2.0)

    np.testing.assert_allclose(maf, np.array([1 / 3, 0.5, 0.0]))


def test_genomic_inflation_factor_handles_empty_and_valid_cases() -> None:
    assert stats_utils.genomic_inflation_factor(np.array([0, np.nan, -1])) == 1.0

    pvalues = np.array([0.5, 0.2, 0.1])
    chi2 = stats.chi2.ppf(1 - pvalues, df=1)
    expected_lambda = np.median(chi2) / stats.chi2.ppf(0.5, df=1)

    assert stats_utils.genomic_inflation_factor(pvalues) == pytest.approx(expected_lambda)


def test_qq_compatible_genomic_inflation_factor_without_subsampling() -> None:
    pvalues = np.array([0.5, 0.2, 0.1])
    expected = stats_utils.genomic_inflation_factor(pvalues)

    observed, is_approx = stats_utils.qq_compatible_genomic_inflation_factor(
        pvalues,
        max_sample_size=10,
        random_seed=123,
    )

    assert is_approx is False
    assert observed == pytest.approx(expected)


def test_qq_compatible_genomic_inflation_factor_with_subsampling() -> None:
    pvalues = np.linspace(1e-6, 1.0, 20)
    rng = np.random.default_rng(7)
    sample = rng.choice(pvalues, size=5, replace=False)
    expected = stats_utils.genomic_inflation_factor(sample)

    observed, is_approx = stats_utils.qq_compatible_genomic_inflation_factor(
        pvalues,
        max_sample_size=5,
        random_seed=7,
    )

    assert is_approx is True
    assert observed == pytest.approx(expected)


def test_qq_plot_data_filters_invalid_and_orders() -> None:
    pvalues = np.array([0.1, 0.5, np.nan, 0.0, -1.0])

    expected, observed = stats_utils.qq_plot_data(pvalues)

    np.testing.assert_allclose(expected, np.array([1 / 3, 2 / 3]))
    np.testing.assert_allclose(observed, np.array([0.1, 0.5]))
