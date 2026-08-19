import numpy as np
import pytest

from panicle.matrix import pca
from panicle.matrix.pca import PANICLE_PCA, PANICLE_PCA_SVD, PANICLE_PCA_genotype
from panicle.utils.data_types import GenotypeMatrix, KinshipMatrix


def test_panicle_pca_requires_input() -> None:
    with pytest.raises(ValueError):
        PANICLE_PCA()


def test_panicle_pca_genotype_sampling_and_batching(monkeypatch, capsys) -> None:
    monkeypatch.setattr(pca, "PCA_MARKER_SAMPLE_THRESHOLD", 2)
    monkeypatch.setattr(pca, "PCA_MARKER_SAMPLE_SIZE", 3)

    genotype = np.array(
        [
            [0.0, 2.0, 1.0, 3.0, 5.0, 4.0],
            [1.0, 0.0, 2.0, 1.0, 3.0, 2.0],
            [2.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [3.0, 3.0, 3.0, 2.0, 0.0, 1.0],
        ]
    )

    pcs = PANICLE_PCA_genotype(
        genotype, pcs_keep=2, maxLine=2, verbose=True
    )

    captured = capsys.readouterr()
    assert "using 3 markers" in captured.out
    assert "Processing batch" in captured.out

    rng = np.random.default_rng(pca.PCA_MARKER_SAMPLE_SEED)
    sample_indices = np.sort(rng.choice(genotype.shape[1], size=3, replace=False))
    subset = genotype[:, sample_indices].astype(np.float32)
    subset -= subset.mean(axis=0)
    covariance = subset @ subset.T / subset.shape[1]
    eigenvals, eigenvecs = np.linalg.eigh(covariance.astype(np.float64))
    eigenvecs = eigenvecs[:, np.argsort(eigenvals)[::-1]][:, : pcs.shape[1]]

    for i in range(pcs.shape[1]):
        alignment = np.abs(np.dot(pcs[:, i], eigenvecs[:, i]))
        assert alignment == pytest.approx(1.0, abs=1e-5)


def test_panicle_pca_genotype_filters_zero_eigenvalues() -> None:
    genotype = np.ones((3, 4), dtype=float)

    pcs = PANICLE_PCA_genotype(genotype, pcs_keep=3, verbose=False)

    assert pcs.shape == (3, 0)


def test_panicle_pca_svd_handles_centering_options() -> None:
    genotype_array = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    genotype_matrix = GenotypeMatrix(genotype_array.copy())

    pcs_centered, var_centered = PANICLE_PCA_SVD(
        genotype_matrix, pcs_keep=2, center=True, verbose=False
    )
    np.testing.assert_allclose(pcs_centered.T @ pcs_centered, np.eye(2), atol=1e-6)
    assert var_centered.shape == (2,)
    assert var_centered.sum() == pytest.approx(1.0)

    pcs_raw, var_raw = PANICLE_PCA_SVD(
        genotype_array, pcs_keep=1, center=False, verbose=False
    )
    assert pcs_raw.shape == (3, 1)
    assert var_raw.shape == (1,)
    assert var_raw[0] > 0


def test_panicle_pca_kinship_accepts_class_verbose(capsys) -> None:
    kinship_array = np.array([[1.0, 0.2], [0.2, 1.0]])
    kinship = KinshipMatrix(kinship_array)

    pcs = PANICLE_PCA(K=kinship, pcs_keep=1, verbose=True)

    captured = capsys.readouterr()
    assert "Performing PCA on kinship matrix" in captured.out
    assert pcs.shape == (2, 1)
    np.testing.assert_allclose(pcs.T @ pcs, np.eye(1), atol=1e-6)


def test_panicle_pca_genotype_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        PANICLE_PCA_genotype("not an array", verbose=False)


def test_panicle_pca_svd_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        PANICLE_PCA_SVD("bad", verbose=False)


def test_validate_pca_results_reports_issues() -> None:
    is_valid, errors = pca.validate_pca_results(np.array([1.0, 2.0]))
    assert not is_valid
    assert any("2D" in err for err in errors)

    pcs_with_nan = np.array([[1.0, 0.0], [0.0, np.nan]])
    is_valid, errors = pca.validate_pca_results(pcs_with_nan)
    assert not is_valid
    assert any("NaN" in err for err in errors)


def test_pca_on_materialized_rows_matches_lazy_view() -> None:
    rng = np.random.default_rng(0)
    full = rng.integers(0, 3, size=(25, 80), dtype=np.int8)
    rows = np.array([0, 2, 5, 9, 11, 18, 24], dtype=np.int64)
    parent = GenotypeMatrix(full, is_imputed=True, precompute_alleles=False)
    lazy = parent.subset_individuals(rows, materialize=False)
    eager = parent.subset_individuals(rows, materialize=True)
    pcs_lazy = PANICLE_PCA(M=lazy, pcs_keep=3, verbose=False)
    pcs_eager = PANICLE_PCA(M=eager, pcs_keep=3, verbose=False)
    np.testing.assert_array_equal(pcs_lazy, pcs_eager)
