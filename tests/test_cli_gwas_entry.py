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
