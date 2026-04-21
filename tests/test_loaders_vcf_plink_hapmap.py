import os
import sys
import types
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panicle.data import load_genotype_vcf, load_genotype_plink, load_genotype_hapmap


def test_load_genotype_vcf_cyvcf2_stub(monkeypatch, tmp_path) -> None:
    seen_threads = []

    class FakeGenotype:
        def __init__(self, arr):
            self._arr = arr

        def array(self):
            return self._arr

    class FakeVariant:
        def __init__(self, chrom, pos, vid, ref, alts, arr):
            self.CHROM = chrom
            self.POS = pos
            self.ID = vid
            self.REF = ref
            self.ALT = alts
            self.genotype = FakeGenotype(arr)

    class FakeVCF:
        def __init__(self, path, threads=1):
            seen_threads.append(threads)
            self.samples = ["S1", "S2"]
            self._vars = [
                FakeVariant("1", 100, "rs1", "A", ["G"], np.array([[0, 0, 0], [0, 1, 0]])),
                FakeVariant("1", 200, "rs2", "C", ["T"], np.array([[1, 1, 0], [-1, -1, 0]])),
                FakeVariant("1", 300, "rs3", "G", ["T", "C"], np.array([[1, 2, 0], [0, 2, 0]])),
                FakeVariant("1", 400, "indel1", "A", ["AT"], np.array([[0, 1, 0], [0, 0, 0]])),
            ]

        def __iter__(self):
            return iter(self._vars)

    fake_module = types.SimpleNamespace(VCF=FakeVCF)
    monkeypatch.setitem(sys.modules, "cyvcf2", fake_module)

    vcf_path = tmp_path / "dummy.vcf"
    vcf_path.write_text("", encoding="utf-8")

    geno, ids, geno_map = load_genotype_vcf.load_genotype_vcf(
        vcf_path,
        backend="cyvcf2",
        include_indels=False,
        split_multiallelic=True,
        drop_monomorphic=False,
        return_pandas=True,
        threads=7,
    )

    assert seen_threads == [7]
    assert ids == ["S1", "S2"]
    assert geno.shape == (2, 3)  # rs1, rs2, rs3 (ALT2 kept, ALT1 invalidated)
    np.testing.assert_array_equal(
        geno,
        np.array(
            [
                [0, 2, 1],  # S1
                [1, 2, 1],  # S2
            ],
            dtype=np.int8,
        ),
    )
    assert list(geno_map["SNP"]) == ["rs1", "rs2", "rs3"]
    assert geno_map["ALT"].tolist() == ["G", "T", "C"]


def test_load_genotype_vcf_prefers_cache(tmp_path, monkeypatch) -> None:
    vcf_path = tmp_path / "cacheme.vcf"
    vcf_path.write_text("", encoding="utf-8")

    geno = np.array([[0, 1], [1, 0]], dtype=np.int8)
    geno_cache = vcf_path.with_suffix(".vcf.panicle.v2.geno.npy")
    np.save(geno_cache, geno)
    ind_cache = vcf_path.with_suffix(".vcf.panicle.v2.ind.txt")
    ind_cache.write_text("S1\nS2\n", encoding="utf-8")
    map_cache = vcf_path.with_suffix(".vcf.panicle.v2.map.csv")
    pd.DataFrame({"SNP": ["rs1", "rs2"], "CHROM": ["1", "1"], "POS": [10, 20]}).to_csv(map_cache, index=False)

    # Ensure cache is strictly newer than VCF; equal mtimes should miss cache.
    newer = time.time() + 10
    for f in (geno_cache, ind_cache, map_cache):
        f.touch()
        os.utime(f, (newer, newer))

    out_geno, ids, map_df = load_genotype_vcf.load_genotype_vcf(vcf_path, backend="builtin", include_indels=True, split_multiallelic=False)

    assert ids == ["S1", "S2"]
    np.testing.assert_array_equal(out_geno, geno)
    assert getattr(map_df, "attrs", {}).get("is_imputed") is True


def test_load_genotype_vcf_bcf_requires_cyvcf2(tmp_path, monkeypatch) -> None:
    bcf_path = tmp_path / "test.bcf"
    bcf_path.write_bytes(b"BCF")

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "cyvcf2":
            raise ImportError("no cyvcf2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError):
        load_genotype_vcf.load_genotype_vcf(bcf_path, backend="auto")


def test_load_genotype_plink_monkeypatched(monkeypatch) -> None:
    def fake_open_bed(path):
        class FakeBed:
            def read(self):
                return np.array(
                    [
                        [0, 2, 2],
                        [2, 2, 1],
                    ],
                    dtype=np.int16,
                )

        return FakeBed()

    fake_module = types.SimpleNamespace(open_bed=fake_open_bed)
    monkeypatch.setitem(sys.modules, "bed_reader", fake_module)

    monkeypatch.setattr(
        load_genotype_plink, "_resolve_plink_paths", lambda prefix_or_bed, bim, fam: (Path("file.bed"), Path("file.bim"), Path("file.fam"))
    )
    monkeypatch.setattr(load_genotype_plink, "_read_fam_ids", lambda fam_path: ["I1", "I2"])
    map_df = pd.DataFrame({"SNP": ["s1", "s2", "s3"], "CHROM": ["1", "1", "1"], "POS": [10, 20, 30], "REF": ["A", "A", "A"], "ALT": ["G", "G", "G"]})
    monkeypatch.setattr(load_genotype_plink, "_read_bim_map", lambda bim_path, return_pandas=True: map_df.copy())

    geno, ids, geno_map = load_genotype_plink.load_genotype_plink("file.bed", drop_monomorphic=True, min_maf=0.0, max_missing=1.0, return_pandas=True)

    assert ids == ["I1", "I2"]
    # Column 2 is monomorphic (all 2) and should be dropped when drop_monomorphic is True
    np.testing.assert_array_equal(geno, np.array([[0, 2], [2, 1]], dtype=np.int8))
    assert list(geno_map["SNP"]) == ["s1", "s3"]


def test_load_genotype_hapmap_filters_and_codes(tmp_path) -> None:
    hmp_path = tmp_path / "toy.hmp.txt"
    hmp_path.write_text(
        "\t".join(
            [
                "rs#", "alleles", "chrom", "pos", "strand", "assembly#", "center", "protLSID", "assayLSID", "panelLSID", "QCcode", "S1", "S2"
            ]
        )
        + "\n"
        + "snp1\tA/C\t1\t100\t+\tNA\tNA\tNA\tNA\tNA\tNA\tA\tC\n"
        + "snp2\tA/T\t1\t200\t+\tNA\tNA\tNA\tNA\tNA\tNA\tAT\tAA\n"
        + "snp3\tG/C\t1\t300\t+\tNA\tNA\tNA\tNA\tNA\tNA\tN\tC\n",
        encoding="utf-8",
    )

    geno, ids, geno_map = load_genotype_hapmap.load_genotype_hapmap(
        hmp_path,
        include_indels=False,
        drop_monomorphic=False,
        min_maf=0.0,
        return_pandas=True,
    )

    assert ids == ["S1", "S2"]
    np.testing.assert_array_equal(geno, np.array([[0, 1, 2], [2, 0, 2]], dtype=np.int8))
    assert list(geno_map["SNP"]) == ["snp1", "snp2", "snp3"]
    assert getattr(geno_map, "attrs", {}).get("is_imputed") is True


def test_load_genotype_hapmap_applies_missingness_and_indel_filter(tmp_path) -> None:
    hmp_path = tmp_path / "filter.hmp.txt"
    hmp_path.write_text(
        "\t".join(_ for _ in ["rs#", "alleles", "chrom", "pos", "strand", "assembly#", "center", "protLSID", "assayLSID", "panelLSID", "QCcode", "S1", "S2"])
        + "\n"
        + "indel\tA/AT\t1\t100\t+\tNA\tNA\tNA\tNA\tNA\tNA\tA\tAT\n"
        + "missing\tA/C\t1\t200\t+\tNA\tNA\tNA\tNA\tNA\tNA\tN\tN\n"
        + "maf_drop\tA/C\t1\t300\t+\tNA\tNA\tNA\tNA\tNA\tNA\tA\tA\n",
        encoding="utf-8",
    )

    geno, ids, geno_map = load_genotype_hapmap.load_genotype_hapmap(
        hmp_path,
        include_indels=False,
        drop_monomorphic=True,
        max_missing=0.5,
        min_maf=0.1,
        return_pandas=True,
    )

    # Only maf_drop filtered as monomorphic; missing dropped for missingness; indel skipped
    assert geno.shape[1] == 0 or geno.size == 0
