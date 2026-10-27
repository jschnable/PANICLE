"""Tests for the packaged panicle-gwas CLI entry point."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from panicle import __version__
from panicle.cli import gwas as cli_gwas
from panicle.cli.utils import parse_args


def test_parse_args_version_exits(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert "panicle-gwas" in out


def test_normalize_methods_and_outputs() -> None:
    assert cli_gwas.normalize_methods(["glm", "FarmCPU", "resampling"]) == [
        "GLM",
        "FARMCPU",
        "FarmCPUResampling",
    ]
    assert cli_gwas.normalize_outputs(["manhattan,qq", "manhattan"]) == [
        "manhattan",
        "qq",
    ]


def test_main_help_returns_via_system_exit() -> None:
    with pytest.raises(SystemExit) as exc:
        cli_gwas.main(["--help"])
    assert exc.value.code == 0


def test_python_m_panicle_version() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "panicle", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert __version__ in proc.stdout


def test_entry_point_module_path_resolves() -> None:
    """pyproject entry point target must import and be callable."""
    mod = importlib.import_module("panicle.cli.gwas")
    assert callable(mod.main)


def test_main_runs_minimal_glm(tmp_path: Path) -> None:
    """End-to-end CLI smoke test on tiny synthetic CSV + map."""
    phe = tmp_path / "phe.csv"
    geno = tmp_path / "geno.csv"
    gmap = tmp_path / "map.csv"
    out = tmp_path / "out"

    phe.write_text(
        "ID,Trait\n"
        "I1,1.0\n"
        "I2,2.0\n"
        "I3,1.5\n"
        "I4,2.5\n"
        "I5,1.2\n"
        "I6,2.1\n"
        "I7,1.8\n"
        "I8,2.2\n"
        "I9,1.1\n"
        "I10,2.4\n"
        "I11,1.3\n"
        "I12,2.0\n",
        encoding="utf-8",
    )
    geno.write_text(
        "ID,m1,m2,m3,m4,m5\n"
        "I1,0,1,2,0,1\n"
        "I2,1,1,0,2,1\n"
        "I3,0,2,1,0,2\n"
        "I4,2,1,0,1,0\n"
        "I5,0,0,2,1,1\n"
        "I6,1,2,1,0,2\n"
        "I7,2,0,0,2,1\n"
        "I8,0,1,1,1,0\n"
        "I9,1,0,2,0,2\n"
        "I10,2,2,0,1,1\n"
        "I11,0,1,1,2,0\n"
        "I12,1,2,2,0,1\n",
        encoding="utf-8",
    )
    gmap.write_text(
        "SNP,CHROM,POS\n"
        "m1,1,100\n"
        "m2,1,200\n"
        "m3,2,50\n"
        "m4,2,150\n"
        "m5,2,250\n",
        encoding="utf-8",
    )

    rc = cli_gwas.main(
        [
            "--phenotype",
            str(phe),
            "--genotype",
            str(geno),
            "--map",
            str(gmap),
            "--format",
            "csv",
            "--methods",
            "GLM",
            "--n-pcs",
            "0",
            "--min-mac",
            "0",
            "--outputs",
            "all_marker_pvalues",
            "significant_marker_pvalues",
            "--outputdir",
            str(out),
        ]
    )
    assert rc == 0
    assert (out / "GWAS_summary_by_traits_methods.csv").exists()
    # At least one full-results CSV for the trait
    result_csvs = list(out.glob("GWAS_*_all_results.csv"))
    assert result_csvs


def _write_stream_inputs(tmp_path: Path, n: int = 40, m: int = 24, T: int = 5):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(21)
    ids = [f"I{i}" for i in range(n)]
    geno = rng.integers(0, 3, size=(n, m))
    phe = pd.DataFrame(rng.normal(size=(n, T)), columns=[f"gene{t}" for t in range(T)])
    phe["gene0"] += 2.0 * geno[:, 2]
    phe.loc[3, "gene1"] = np.nan  # one sample dropped from the shared set
    phe.insert(0, "ID", ids)
    gdf = pd.DataFrame(geno, columns=[f"m{j}" for j in range(m)])
    gdf.insert(0, "ID", ids)
    gmap = pd.DataFrame({"SNP": [f"m{j}" for j in range(m)], "CHROM": [1] * (m // 2) + [2] * (m - m // 2), "POS": range(100, 100 + m)})
    phe.to_csv(tmp_path / "phe.csv", index=False)
    gdf.to_csv(tmp_path / "geno.csv", index=False)
    gmap.to_csv(tmp_path / "map.csv", index=False)
    return T, m


def test_main_stream_results_hits(tmp_path: Path, capsys) -> None:
    import pandas as pd

    T, m = _write_stream_inputs(tmp_path)
    out = tmp_path / "out"
    rc = cli_gwas.main(
        [
            "--phenotype", str(tmp_path / "phe.csv"),
            "--genotype", str(tmp_path / "geno.csv"),
            "--map", str(tmp_path / "map.csv"),
            "--format", "csv",
            "--n-pcs", "0",
            "--min-mac", "0",
            "--stream-results", "hits",
            "--stream-p-threshold", "1e-3",
            "--outputdir", str(out),
        ]
    )
    assert rc == 0
    for fname in ("hits.csv", "trait_summary.csv", "markers.csv", "metadata.json"):
        assert (out / fname).exists()
    assert not (out / "effects.npy").exists()
    assert not list(out.glob("GWAS_*_all_results.csv"))
    hits = pd.read_csv(out / "hits.csv")
    assert list(hits.columns) == ["Trait", "MARKER", "CHROM", "POS", "Effect", "SE", "P_value"]
    assert "gene0" in set(hits["Trait"]) and "m2" in set(hits.loc[hits["Trait"] == "gene0", "MARKER"])
    summary = pd.read_csv(out / "trait_summary.csv")
    assert list(summary["Trait"]) == [f"gene{t}" for t in range(T)]
    assert summary.set_index("Trait").loc["gene0", "top_MARKER"] == "m2"
    assert len(pd.read_csv(out / "markers.csv")) == m
    assert "Excluded 1 sample(s)" in capsys.readouterr().out


def test_main_stream_results_all_reopens(tmp_path: Path) -> None:
    import numpy as np
    from panicle.association.multi_trait_store import MultiTraitResultStore

    T, m = _write_stream_inputs(tmp_path)
    out = tmp_path / "out_all"
    rc = cli_gwas.main(
        [
            "--phenotype", str(tmp_path / "phe.csv"),
            "--genotype", str(tmp_path / "geno.csv"),
            "--map", str(tmp_path / "map.csv"),
            "--format", "csv",
            "--n-pcs", "1",
            "--min-mac", "0",
            "--methods", "MLM",
            "--stream-results", "all",
            "--no-lrt-refinement",
            "--stream-dtype", "float64",
            "--trait-block-size", "2",
            "--outputdir", str(out),
        ]
    )
    assert rc == 0
    store = MultiTraitResultStore.open(out)
    assert store.keep == "all" and store.pvalues.shape == (T, m)
    assert store.effects.dtype == np.float64
    assert list(store.markers["MARKER"]) == [f"m{j}" for j in range(m)]
    res = store["gene0"]
    assert res.pvalues.argmin() == 2
    assert np.all(np.isfinite(res.pvalues))

