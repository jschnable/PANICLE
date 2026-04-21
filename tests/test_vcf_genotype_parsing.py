"""Unit tests for GT parsing helpers in load_genotype_vcf."""

from typing import Tuple

import numpy as np
import pytest

from panicle.data.load_genotype_vcf import (
    _DynamicInt8MatrixWriter,
    MISSING,
    _code_dosage_biallelic,
    _code_dosage_split,
    _decode_biallelic_gt,
    _split_gt_tokens,
)


@pytest.mark.parametrize(
    "gt,expected",
    [
        ("0/0", ("0", "0")),
        ("0|1", ("0", "1")),
        ("1/1/1", ("1", "1", "1")),
    ],
)
def test_split_gt_tokens_returns_cached_tuple(gt: str, expected: Tuple[str, ...]) -> None:
    first = _split_gt_tokens(gt)
    second = _split_gt_tokens(gt)
    assert first == expected
    assert second == expected
    # Cached results should reuse the same tuple object
    assert first is second


@pytest.mark.parametrize("gt", [None, ".", "./.", ".|.", "0"])
def test_split_gt_tokens_handles_missing(gt) -> None:
    assert _split_gt_tokens(gt) is None


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (("0", "0"), (0, 2)),
        (("0", "1"), (1, 2)),
        (("1", "1"), (2, 2)),
        (("1", "1", "0"), (2, 3)),
        (("1", "1", "1"), (3, 3)),
    ],
)
def test_code_dosage_biallelic_valid(tokens: Tuple[str, ...], expected: Tuple[int, int]) -> None:
    assert _code_dosage_biallelic(tokens) == expected


@pytest.mark.parametrize(
    "tokens",
    [
        ("1", "."),
        ("2", "0"),
        ("A", "1"),
    ],
)
def test_code_dosage_biallelic_invalid(tokens: Tuple[str, ...]) -> None:
    assert _code_dosage_biallelic(tokens) == (MISSING, 0)


def test_code_dosage_biallelic_cache_accepts_lists() -> None:
    tokens_list = ["0", "1"]
    first = _code_dosage_biallelic(tokens_list)
    second = _code_dosage_biallelic(tuple(tokens_list))
    assert first == (1, 2)
    assert first == second


@pytest.mark.parametrize(
    "gt,expected",
    [
        ("0/0", (0, 2)),
        ("0|1", (1, 2)),
        ("1/1/1", (3, 3)),
        ("0", (0, 1)),
    ],
)
def test_decode_biallelic_gt_valid(gt: str, expected: Tuple[int, int]) -> None:
    assert _decode_biallelic_gt(gt) == expected
    # repeated call hits cache
    assert _decode_biallelic_gt(gt) == expected


@pytest.mark.parametrize("gt", [None, "./.", "0/2", "A/1", ""])
def test_decode_biallelic_gt_invalid(gt) -> None:
    assert _decode_biallelic_gt(gt) == (MISSING, 0)


@pytest.mark.parametrize(
    "tokens,alt_index,expected",
    [
        (("0", "2"), 2, (1, 2)),
        (("2", "2"), 2, (2, 2)),
        (("0", "1"), 2, (MISSING, 0)),
        (("3", "2", "0"), 3, (MISSING, 0)),
    ],
)
def test_code_dosage_split(tokens: Tuple[str, ...], alt_index: int, expected: Tuple[int, int]) -> None:
    assert _code_dosage_split(tokens, alt_index) == expected


def test_dynamic_int8_writer_preserves_sample_major_shape() -> None:
    writer = _DynamicInt8MatrixWriter(3, initial_capacity=1)
    writer.append(np.array([0, 1, 2], dtype=np.int8))
    writer.append(np.array([2, 1, 0], dtype=np.int8))

    result = writer.finalize()

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [0, 2],
                [1, 1],
                [2, 0],
            ],
            dtype=np.int8,
        ),
    )
    assert result.flags.c_contiguous
