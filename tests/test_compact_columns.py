"""Run-length int8 column compaction vs fancy-index gather."""

from __future__ import annotations

import numpy as np
import pytest

from panicle.utils.compact import (
    can_compact_int8_columns,
    compact_columns_int8,
    int8_to_float32,
    keep_indices_are_strictly_increasing,
    runs_from_keep_indices,
)
from panicle.utils.data_types import GenotypeMatrix


def test_runs_from_keep_indices_empty_and_single() -> None:
    starts, lens = runs_from_keep_indices(np.array([], dtype=np.int64))
    assert starts.size == 0
    assert lens.size == 0

    starts, lens = runs_from_keep_indices(np.array([7], dtype=np.int64))
    np.testing.assert_array_equal(starts, [7])
    np.testing.assert_array_equal(lens, [1])


def test_runs_from_keep_indices_clusters() -> None:
    keep = np.array([0, 1, 2, 5, 8, 9, 10, 11], dtype=np.int64)
    starts, lens = runs_from_keep_indices(keep)
    np.testing.assert_array_equal(starts, [0, 5, 8])
    np.testing.assert_array_equal(lens, [3, 1, 4])
    assert int(lens.sum()) == keep.size


def test_compact_matches_fancy_index_random_holes() -> None:
    rng = np.random.default_rng(0)
    src = rng.integers(0, 3, size=(40, 2000), dtype=np.int8)
    keep = np.flatnonzero(rng.random(2000) > 0.06).astype(np.int64)
    got = compact_columns_int8(src, keep)
    expected = np.ascontiguousarray(src[:, keep])
    np.testing.assert_array_equal(got, expected)
    assert got.flags.c_contiguous
    assert got.base is None or got.base is not src


def test_compact_all_kept_is_one_run() -> None:
    src = np.arange(6 * 10, dtype=np.int8).reshape(6, 10)
    keep = np.arange(10, dtype=np.int64)
    starts, lens = runs_from_keep_indices(keep)
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_array_equal(lens, [10])
    np.testing.assert_array_equal(compact_columns_int8(src, keep), src)


def test_compact_alternating_singletons() -> None:
    src = np.arange(4 * 12, dtype=np.int8).reshape(4, 12)
    keep = np.arange(0, 12, 2, dtype=np.int64)
    starts, lens = runs_from_keep_indices(keep)
    np.testing.assert_array_equal(starts, keep)
    np.testing.assert_array_equal(lens, np.ones(keep.size, dtype=np.int64))
    np.testing.assert_array_equal(compact_columns_int8(src, keep), src[:, keep])


def test_compact_hole_at_first_and_last_column() -> None:
    src = np.arange(5 * 8, dtype=np.int8).reshape(5, 8)
    keep = np.arange(1, 7, dtype=np.int64)  # drop 0 and 7
    starts, lens = runs_from_keep_indices(keep)
    np.testing.assert_array_equal(starts, [1])
    np.testing.assert_array_equal(lens, [6])
    np.testing.assert_array_equal(compact_columns_int8(src, keep), src[:, keep])

    keep_first_only = np.arange(1, 8, dtype=np.int64)
    starts, lens = runs_from_keep_indices(keep_first_only)
    np.testing.assert_array_equal(starts, [1])
    np.testing.assert_array_equal(lens, [7])

    keep_last_only = np.arange(0, 7, dtype=np.int64)
    starts, lens = runs_from_keep_indices(keep_last_only)
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_array_equal(lens, [7])


def test_compact_empty_keep() -> None:
    src = np.arange(3 * 5, dtype=np.int8).reshape(3, 5)
    keep = np.array([], dtype=np.int64)
    starts, lens = runs_from_keep_indices(keep)
    assert starts.size == 0 and lens.size == 0
    got = compact_columns_int8(src, keep)
    assert got.shape == (3, 0)
    assert got.dtype == np.int8

    gm = GenotypeMatrix(src, is_imputed=True, precompute_alleles=False)
    sub = gm.subset_markers(keep)
    assert sub.shape == (3, 0)


def test_compact_rejects_non_increasing() -> None:
    src = np.zeros((2, 5), dtype=np.int8)
    with pytest.raises(ValueError, match="strictly increasing"):
        compact_columns_int8(src, np.array([0, 2, 1], dtype=np.int64))
    assert not keep_indices_are_strictly_increasing(np.array([1, 1, 2]))
    assert not can_compact_int8_columns(src, np.array([2, 0], dtype=np.int64))


def test_subset_markers_matches_fancy_index() -> None:
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 3, size=(20, 80), dtype=np.int8)
    gm = GenotypeMatrix(arr, is_imputed=True, precompute_alleles=False)
    keep = np.flatnonzero(rng.random(80) > 0.1).astype(np.int64)
    sub = gm.subset_markers(keep)
    np.testing.assert_array_equal(sub.to_numpy(), arr[:, keep])
    assert sub._storage.flags.c_contiguous
    assert sub._storage.shape == (20, keep.size)


def test_subset_markers_non_monotonic_still_works() -> None:
    arr = np.arange(4 * 6, dtype=np.int8).reshape(4, 6)
    gm = GenotypeMatrix(arr, is_imputed=True, precompute_alleles=False)
    sub = gm.subset_markers(np.array([5, 1, 3], dtype=np.int64))
    np.testing.assert_array_equal(sub.to_numpy(), arr[:, [5, 1, 3]])


def test_compact_row_idx_matches_ix_() -> None:
    rng = np.random.default_rng(4)
    src = rng.integers(0, 3, size=(20, 80), dtype=np.int8)
    keep = np.flatnonzero(rng.random(80) > 0.15).astype(np.int64)
    rows = np.array([0, 3, 7, 18, 2], dtype=np.int64)
    got = compact_columns_int8(src, keep, row_idx=rows)
    expected = np.ascontiguousarray(src[np.ix_(rows, keep)])
    np.testing.assert_array_equal(got, expected)


def test_subset_markers_on_row_view_does_not_fancy_index() -> None:
    rng = np.random.default_rng(5)
    src = rng.integers(0, 3, size=(15, 40), dtype=np.int8)
    gm = GenotypeMatrix(src, is_imputed=True, precompute_alleles=False)
    rows = np.array([1, 4, 9, 14], dtype=np.int64)
    view = gm.subset_individuals(rows, materialize=False)
    keep = np.array([0, 2, 5, 9, 20], dtype=np.int64)
    packed = view.subset_markers(keep)
    np.testing.assert_array_equal(packed.to_numpy(), src[np.ix_(rows, keep)])
    assert packed.n_individuals == 4
    assert packed.n_markers == 5


def test_mac_on_lazy_row_view_matches_materialized() -> None:
    from panicle.utils.stats import compute_mac_keep_indices

    rng = np.random.default_rng(6)
    src = rng.integers(0, 3, size=(30, 50), dtype=np.int8)
    src[:, 1] = 0
    src[0, 1] = 2
    gm = GenotypeMatrix(src, is_imputed=True, precompute_alleles=False)
    rows = np.arange(5, 28, dtype=np.int64)
    lazy = gm.subset_individuals(rows, materialize=False)
    eager = gm.subset_individuals(rows, materialize=True)
    np.testing.assert_array_equal(
        compute_mac_keep_indices(lazy, 5),
        compute_mac_keep_indices(eager, 5),
    )


def test_subset_markers_logs_fancy_index_fallback(caplog) -> None:
    import logging

    arr = np.arange(4 * 6, dtype=np.int8).reshape(4, 6)
    gm = GenotypeMatrix(arr, is_imputed=True, precompute_alleles=False)
    with caplog.at_level(logging.DEBUG, logger="panicle.utils.data_types"):
        gm.subset_markers(np.array([5, 1, 3], dtype=np.int64))
    assert any("fancy index" in rec.message for rec in caplog.records)
    assert any("not strictly increasing" in rec.message for rec in caplog.records)


def test_int8_to_float32_matches_numpy_astype() -> None:
    rng = np.random.default_rng(7)
    src = rng.integers(-9, 3, size=(64, 4096), dtype=np.int8)
    got = int8_to_float32(src)
    np.testing.assert_array_equal(got, src.astype(np.float32))
    assert got.dtype == np.float32
    assert got.flags.c_contiguous

    # Large enough to take the parallel kernel (>= 8e6 elements).
    big = rng.integers(-9, 3, size=(200, 50_000), dtype=np.int8)
    np.testing.assert_array_equal(int8_to_float32(big), big.astype(np.float32))


def test_int8_to_float32_empty_and_out() -> None:
    src = np.zeros((3, 0), dtype=np.int8)
    got = int8_to_float32(src)
    assert got.shape == (3, 0)
    assert got.dtype == np.float32

    src = np.array([[0, 1, 2], [-9, 0, 1]], dtype=np.int8)
    out = np.empty_like(src, dtype=np.float32)
    ret = int8_to_float32(src, out=out)
    assert ret is out
    np.testing.assert_array_equal(out, src.astype(np.float32))


def test_get_columns_float32_matches_astype() -> None:
    rng = np.random.default_rng(8)
    arr = rng.integers(0, 3, size=(12, 40), dtype=np.int8)
    gm = GenotypeMatrix(arr, is_imputed=True, precompute_alleles=False)
    idx = np.arange(5, 25, dtype=np.int64)
    got = gm.get_columns(idx, dtype=np.float32)
    np.testing.assert_array_equal(got, arr[:, idx].astype(np.float32))
    batch = gm.get_batch(5, 25, dtype=np.float32)
    np.testing.assert_array_equal(batch, arr[:, 5:25].astype(np.float32))
