"""On-disk cache for leave-one-group-out Gram objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import panicle.pipelines.gwas as gwas_module
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.matrix.loco_cache import (
    load_loco_kinship,
    loco_cache_digest,
    loco_cache_path,
    save_loco_kinship,
)
from panicle.pipelines.gwas import GWASPipeline
from panicle.utils.data_types import GenotypeMatrix


def _tiny_loco():
    rng = np.random.default_rng(3)
    geno = rng.integers(0, 3, size=(12, 24), dtype=np.int8)
    map_df = pd.DataFrame({
        "SNP": [f"s{i}" for i in range(24)],
        "CHROM": ["1"] * 12 + ["2"] * 12,
        "POS": np.arange(24) * 10,
    })
    return PANICLE_K_VanRaden_LOCO(
        GenotypeMatrix(geno, is_imputed=True, precompute_alleles=False),
        map_df,
        maxLine=5,
        verbose=False,
    )


def test_loco_cache_roundtrip(tmp_path: Path) -> None:
    loco = _tiny_loco()
    path = tmp_path / "loco.npz"
    save_loco_kinship(path, loco)
    loaded = load_loco_kinship(path)
    assert loaded is not None
    assert loaded.chromosomes == loco.chromosomes
    np.testing.assert_array_equal(loaded.get_full().to_numpy(), loco.get_full().to_numpy())
    for chrom in loco.chromosomes:
        np.testing.assert_array_equal(
            loaded.get_loco(chrom).to_numpy(),
            loco.get_loco(chrom).to_numpy(),
        )


def test_loco_cache_digest_changes_with_rows_and_keep() -> None:
    rows = np.arange(10, dtype=np.int64)
    keep = np.array([0, 1, 3, 5], dtype=np.int64)
    chroms = ["1", "2"]
    base = loco_cache_digest(rows, keep, chroms, n_markers=4, max_line=5000)
    assert base == loco_cache_digest(rows, keep, chroms, n_markers=4, max_line=5000)
    assert base != loco_cache_digest(rows[::-1], keep, chroms, n_markers=4, max_line=5000)
    assert base != loco_cache_digest(rows, keep[:-1], chroms, n_markers=3, max_line=5000)
    assert base != loco_cache_digest(rows, None, chroms, n_markers=4, max_line=5000)
    assert base != loco_cache_digest(rows, keep, ["1"], n_markers=4, max_line=5000)
    # Batch width is not part of the v2 digest: the exact Gram does not depend on it.
    assert base == loco_cache_digest(rows, keep, chroms, n_markers=4, max_line=1000)


def test_loco_cache_path_sidecar_dir() -> None:
    path = loco_cache_path("/tmp/panel.vcf.gz", "abc123")
    assert path.name == "abc123.npz"
    assert path.parent.name.endswith(".panicle.v2.loco")


def test_load_loco_kinship_missing_returns_none(tmp_path: Path) -> None:
    assert load_loco_kinship(tmp_path / "nope.npz") is None


def test_pipeline_reuses_disk_loco_cache(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(11)
    n_samples, n_markers = 20, 40
    ids = [f"S{i:02d}" for i in range(n_samples)]
    pheno = pd.DataFrame({"ID": ids, "Height": rng.standard_normal(n_samples)})
    geno = pd.DataFrame(rng.integers(0, 3, size=(n_samples, n_markers)), columns=[f"m{i}" for i in range(n_markers)])
    geno.insert(0, "ID", ids)
    gmap = pd.DataFrame({
        "SNP": [f"m{i}" for i in range(n_markers)],
        "CHROM": ["1"] * 20 + ["2"] * 20,
        "POS": np.arange(n_markers) * 100,
    })
    pheno_file = tmp_path / "pheno.csv"
    geno_file = tmp_path / "geno.csv"
    map_file = tmp_path / "map.csv"
    pheno.to_csv(pheno_file, index=False)
    geno.to_csv(geno_file, index=False)
    gmap.to_csv(map_file, index=False)

    calls = {"n": 0}
    real = gwas_module.PANICLE_K_VanRaden_LOCO

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(gwas_module, "PANICLE_K_VanRaden_LOCO", wrapped)

    def _run(out_name: str) -> None:
        pipeline = GWASPipeline(output_dir=str(tmp_path / out_name))
        pipeline.load_data(
            phenotype_file=str(pheno_file),
            genotype_file=str(geno_file),
            map_file=str(map_file),
            trait_columns=["Height"],
            genotype_format="csv",
        )
        pipeline.align_samples()
        pipeline.compute_population_structure(n_pcs=2, calculate_kinship=False)
        pipeline.run_analysis(
            traits=["Height"],
            methods=["MLM"],
            min_mac=0,
            outputs=["significant_marker_pvalues"],
        )

    _run("first")
    assert calls["n"] == 1
    cache_dir = Path(str(geno_file) + ".panicle.v2.loco")
    assert cache_dir.is_dir()
    assert any(cache_dir.glob("*.npz"))

    _run("second")
    assert calls["n"] == 1
