"""Exact 0/1/2 VanRaden Gram: batch-width and thread independence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panicle.matrix.kinship import PANICLE_K_VanRaden
from panicle.matrix.kinship_exact import (
    UncenteredGram,
    can_use_exact_gram,
)
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.utils.data_types import GenotypeMatrix


def _imputed(arr: np.ndarray) -> GenotypeMatrix:
    return GenotypeMatrix(arr, is_imputed=True, precompute_alleles=False)


def test_can_use_exact_gram_imputed_int8() -> None:
    g = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int8)
    assert can_use_exact_gram(_imputed(g)) is True
    assert can_use_exact_gram(GenotypeMatrix(g, is_imputed=False, precompute_alleles=False)) is False
    assert can_use_exact_gram(g) is True
    missing = g.copy()
    missing[0, 0] = -9
    assert can_use_exact_gram(missing) is False


def test_batch_gram_of_012_is_integer() -> None:
    rng = np.random.default_rng(0)
    Z = rng.integers(0, 3, size=(40, 80), dtype=np.int8)
    gram = UncenteredGram(40)
    gram.add_batch(Z)
    np.testing.assert_allclose(gram.S, np.rint(gram.S), atol=0, rtol=0)
    expected = (Z.astype(np.int64) @ Z.astype(np.int64).T).astype(np.float64)
    np.testing.assert_array_equal(gram.S, expected)


def test_vanraden_exact_matches_reference_and_batch_widths() -> None:
    rng = np.random.default_rng(1)
    g = rng.integers(0, 3, size=(25, 120), dtype=np.int8)
    gm = _imputed(g)
    k5 = PANICLE_K_VanRaden(gm, maxLine=5, verbose=False).to_numpy()
    k7 = PANICLE_K_VanRaden(gm, maxLine=7, verbose=False).to_numpy()
    k40 = PANICLE_K_VanRaden(gm, maxLine=40, verbose=False).to_numpy()
    np.testing.assert_array_equal(k5, k7)
    np.testing.assert_array_equal(k5, k40)

    n = g.shape[0]
    mu = g.mean(axis=0)
    Zc = g.astype(np.float64) - mu
    ref = Zc @ Zc.T
    ref /= np.mean(np.diag(ref))
    np.testing.assert_allclose(k5, ref, rtol=1e-12, atol=1e-12)


def test_loco_exact_batch_widths_and_matches_vanraden() -> None:
    rng = np.random.default_rng(2)
    g = rng.integers(0, 3, size=(18, 36), dtype=np.int8)
    chroms = np.array(["1"] * 12 + ["2"] * 12 + ["3"] * 12)
    map_df = pd.DataFrame({
        "SNP": [f"s{i}" for i in range(36)],
        "CHROM": chroms,
        "POS": np.arange(36),
    })
    gm = _imputed(g)
    loco_a = PANICLE_K_VanRaden_LOCO(gm, map_df, maxLine=5, verbose=False)
    loco_b = PANICLE_K_VanRaden_LOCO(gm, map_df, maxLine=11, verbose=False)
    np.testing.assert_array_equal(loco_a.get_full().to_numpy(), loco_b.get_full().to_numpy())
    for chrom in loco_a.chromosomes:
        np.testing.assert_array_equal(
            loco_a.get_loco(chrom).to_numpy(),
            loco_b.get_loco(chrom).to_numpy(),
        )
        keep = chroms != chrom
        ref = PANICLE_K_VanRaden(_imputed(g[:, keep]), maxLine=5, verbose=False).to_numpy()
        np.testing.assert_array_equal(loco_a.get_loco(chrom).to_numpy(), ref)

    full = PANICLE_K_VanRaden(gm, maxLine=8, verbose=False).to_numpy()
    np.testing.assert_array_equal(loco_a.get_full().to_numpy(), full)


def test_exact_gram_independent_of_openblas_threads() -> None:
    threadpoolctl = pytest.importorskip("threadpoolctl")
    rng = np.random.default_rng(4)
    g = rng.integers(0, 3, size=(30, 90), dtype=np.int8)
    gm = _imputed(g)
    with threadpoolctl.threadpool_limits(1, user_api="blas"):
        k1 = PANICLE_K_VanRaden(gm, maxLine=6, verbose=False).to_numpy()
    with threadpoolctl.threadpool_limits(4, user_api="blas"):
        k4 = PANICLE_K_VanRaden(gm, maxLine=6, verbose=False).to_numpy()
    np.testing.assert_array_equal(k1, k4)
