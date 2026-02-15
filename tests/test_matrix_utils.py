import numpy as np
import pytest

from panicle.matrix.kinship import PANICLE_K_IBS, PANICLE_K_VanRaden, validate_kinship_matrix
from panicle.matrix.pca import PANICLE_PCA, PANICLE_PCA_genotype, PANICLE_PCA_kinship
from panicle.utils.data_types import GenotypeMatrix


def test_panicle_pca_kinship_returns_expected_vectors() -> None:
    kinship_matrix = np.array([[2.0, 1.0], [1.0, 2.0]])

    pcs = PANICLE_PCA_kinship(kinship_matrix, pcs_keep=2, verbose=False)

    assert pcs.shape == (2, 2)
    np.testing.assert_allclose(pcs.T @ pcs, np.eye(2), atol=1e-6)
    expected_first = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    np.testing.assert_allclose(np.abs(pcs[:, 0]), expected_first, atol=1e-6)


def test_panicle_pca_genotype_matches_covariance_eigenvectors() -> None:
    genotype = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    covariance = np.zeros((3, 3), dtype=np.float32)
    n_markers = genotype.shape[1]
    max_line = 2
    for start in range(0, n_markers, max_line):
        end = min(start + max_line, n_markers)
        batch = genotype[:, start:end].astype(np.float32)
        batch -= np.mean(batch, axis=0)[np.newaxis, :]
        covariance += batch @ batch.T
    covariance = covariance.astype(np.float64) / n_markers

    eigenvals, eigenvecs = np.linalg.eigh(covariance)
    sort_idx = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, sort_idx]

    pcs = PANICLE_PCA_genotype(genotype, pcs_keep=3, maxLine=max_line, verbose=False)

    assert pcs.shape[0] == genotype.shape[0]
    assert pcs.shape[1] == eigenvecs.shape[1]
    for i in range(pcs.shape[1]):
        alignment = np.abs(np.dot(pcs[:, i], eigenvecs[:, i]))
        assert alignment == pytest.approx(1.0)


def test_panicle_pca_prefers_kinship_when_both_provided() -> None:
    genotype = np.array([[0.0, 1.0], [1.0, 0.0]])
    kinship_matrix = np.array([[1.0, 0.2], [0.2, 1.0]])

    with pytest.warns(UserWarning):
        pcs = PANICLE_PCA(M=genotype, K=kinship_matrix, pcs_keep=1, verbose=False)

    expected = PANICLE_PCA_kinship(kinship_matrix, pcs_keep=1, verbose=False)
    np.testing.assert_allclose(np.abs(pcs[:, 0]), np.abs(expected[:, 0]), atol=1e-6)


def test_panicle_k_ibs_small_matrix_matches_manual_calculation() -> None:
    genotype_array = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int8)
    genotype = GenotypeMatrix(genotype_array)

    kinship = PANICLE_K_IBS(genotype, maxLine=5, verbose=False)

    expected = np.array(
        [
            [1.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0],
        ]
    )
    np.testing.assert_allclose(kinship.to_numpy(), expected)


def test_validate_kinship_matrix_detects_shape_and_psd_issues() -> None:
    is_valid, errors = validate_kinship_matrix(np.ones((2, 3)))
    assert not is_valid
    assert "not square" in errors[0]

    is_valid, errors = validate_kinship_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert not is_valid
    assert any("positive semi-definite" in err for err in errors)

    is_valid, errors = validate_kinship_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert is_valid
    assert errors == []


def test_panicle_k_vanraden_throttles_progress_logging(caplog) -> None:
    rng = np.random.default_rng(5)
    geno = rng.integers(0, 3, size=(6, 80), dtype=np.int8)

    with caplog.at_level("INFO", logger="panicle.matrix.kinship"):
        PANICLE_K_VanRaden(geno, maxLine=2, verbose=True)

    progress_records = [r for r in caplog.records if "Processing batch" in r.message]
    # 80 markers / 2 per batch = 40 batches; logging should be throttled below per-batch spam.
    assert 0 < len(progress_records) < 40
