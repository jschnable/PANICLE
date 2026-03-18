"""Unit tests for harmonised GEC effective marker count utilities."""

from __future__ import annotations

from importlib import util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from panicle.data.loaders import load_genotype_file
from panicle.utils.data_types import GenotypeMap, GenotypeMatrix
from panicle.utils.effective_tests import estimate_effective_tests_from_genotype


def _load_run_gwas_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_GWAS.py"
    spec = util.spec_from_file_location("run_GWAS_cli_effective", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_GWAS.py module for testing")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


RUN_GWAS = _load_run_gwas_module()


def test_estimate_effective_tests_from_genotype_simple_case() -> None:
    """Perfectly correlated SNPs collapse to a single effective test."""
    geno_array = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int8,
    )
    genotype = GenotypeMatrix(geno_array)
    map_df = pd.DataFrame(
        {
            "SNP": ["snp1", "snp2", "snp3"],
            "CHROM": ["1", "1", "1"],
            "POS": [100, 150, 10_000],
        }
    )
    geno_map = GenotypeMap(map_df)

    result = estimate_effective_tests_from_genotype(genotype, geno_map)

    assert result["Me"] == 2
    assert result["total_snps"] == 3
    assert result["per_chromosome"]["1"]["n_snps"] == 3
    assert result["per_chromosome"]["1"]["Me"] == 2
    assert len(result["block_stats"]) >= 1


def test_estimate_effective_tests_parallel_matches_serial() -> None:
    geno_array = np.array(
        [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    genotype = GenotypeMatrix(geno_array)
    map_df = pd.DataFrame(
        {
            "SNP": ["snp1", "snp2", "snp3", "snp4"],
            "CHROM": ["1", "1", "1", "1"],
            "POS": [100, 200, 500, 800],
        }
    )
    geno_map = GenotypeMap(map_df)

    serial = estimate_effective_tests_from_genotype(genotype, geno_map, ncpus=1)
    parallel = estimate_effective_tests_from_genotype(genotype, geno_map, ncpus=2)

    assert parallel["Me"] == serial["Me"]
    assert parallel["total_snps"] == serial["total_snps"]
    assert set(parallel["per_chromosome"]) == set(serial["per_chromosome"])


def test_estimate_effective_tests_rejects_negative_cpu() -> None:
    geno_array = np.array(
        [
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
        ],
        dtype=np.int8,
    )
    genotype = GenotypeMatrix(geno_array)
    map_df = pd.DataFrame(
        {
            "SNP": ["snp1", "snp2"],
            "CHROM": ["1", "1"],
            "POS": [100, 200],
        }
    )
    geno_map = GenotypeMap(map_df)

    with pytest.raises(ValueError, match=">= 0"):
        estimate_effective_tests_from_genotype(genotype, geno_map, ncpus=-1)


def test_load_genotype_file_records_effective_tests(tmp_path: Path) -> None:
    geno_file = tmp_path / "geno.csv"
    geno_file.write_text(
        "ID,snp1,snp2,snp3\n"
        "I1,0,0,0\n"
        "I2,1,1,1\n"
        "I3,0,0,1\n"
        "I4,1,1,0\n",
        encoding="utf-8",
    )

    genotype_matrix, individual_ids, geno_map = load_genotype_file(
        geno_file,
        file_format="csv",
        compute_effective_tests=True,
    )

    assert genotype_matrix.n_markers == 3
    assert len(individual_ids) == 4
    effective_info = geno_map.metadata.get("effective_tests")
    assert effective_info is not None
    assert effective_info["Me"] == 2
    assert effective_info["total_snps"] == 3
    assert effective_info["dropped_monomorphic_total"] == 0


@pytest.mark.skip(reason="load_and_validate_data was refactored into GWASPipeline")
def test_load_and_validate_data_returns_effective_tests(tmp_path: Path) -> None:
    phe_file = tmp_path / "phe.csv"
    phe_file.write_text(
        "ID,Trait\n"
        "I1,1.0\n"
        "I2,2.0\n"
        "I3,3.0\n"
        "I4,4.0\n",
        encoding="utf-8",
    )

    geno_file = tmp_path / "geno.csv"
    geno_file.write_text(
        "ID,snp1,snp2,snp3\n"
        "I1,0,0,0\n"
        "I2,1,1,1\n"
        "I3,0,0,1\n"
        "I4,1,1,0\n",
        encoding="utf-8",
    )

    map_file = tmp_path / "map.csv"
    map_file.write_text(
        "SNP,CHROM,POS\n"
        "snp1,1,100\n"
        "snp2,1,150\n"
        "snp3,1,10000\n",
        encoding="utf-8",
    )

    data = RUN_GWAS.load_and_validate_data(
        phenotype_file=str(phe_file),
        genotype_file=str(geno_file),
        map_file=str(map_file),
        genotype_format="csv",
        loader_kwargs={"compute_effective_tests": True},
    )

    effective_info = data.get("effective_tests")
    assert effective_info is not None
    assert effective_info["Me"] == 2
    assert effective_info["total_snps"] == 3
    assert data["geno_map"].metadata.get("effective_tests") == effective_info


def test_monomorphic_markers_are_dropped(tmp_path: Path) -> None:
    geno_file = tmp_path / "geno.csv"
    geno_file.write_text(
        "ID,snp1,snp2,snp3\n"
        "I1,0,0,0\n"
        "I2,1,1,0\n"
        "I3,0,0,0\n"
        "I4,1,1,0\n",
        encoding="utf-8",
    )

    genotype_matrix, _, geno_map = load_genotype_file(
        geno_file,
        file_format="csv",
        compute_effective_tests=True,
    )

    effective_info = geno_map.metadata["effective_tests"]
    assert effective_info["Me"] == 1
    assert effective_info["total_snps"] == 3
    assert effective_info["dropped_monomorphic_total"] == 1
    chrom_info = effective_info["per_chromosome"]["1"]
    assert chrom_info["monomorphic_dropped"] == 1
    assert chrom_info["n_snps"] == 2
