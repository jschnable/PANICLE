import sys
import pytest

from panicle.cli import utils


def test_normalize_outputs_defaults_and_filters() -> None:
    assert utils.normalize_outputs([]) == list(utils.OUTPUT_CHOICES)
    assert utils.normalize_outputs([" manhattan ", "qq", "unknown"]) == ["manhattan", "qq"]
    # all invalid falls back to defaults
    assert utils.normalize_outputs(["bad"]) == list(utils.OUTPUT_CHOICES)


def _run_parse_args(argv):
    orig = sys.argv
    try:
        sys.argv = argv
        return utils.parse_args()
    finally:
        sys.argv = orig


def test_parse_args_defaults_and_flags(tmp_path) -> None:
    args = _run_parse_args(
        [
            "prog",
            "--phenotype",
            str(tmp_path / "phe.csv"),
            "--genotype",
            str(tmp_path / "geno.csv"),
        ]
    )
    assert args.phenotype.endswith("phe.csv")
    assert args.genotype.endswith("geno.csv")
    assert args.drop_monomorphic is True  # default
    assert args.outputs == list(utils.OUTPUT_CHOICES)
    assert args.outputdir == "./GWAS_results"
    assert args.ncpus == 1
    assert args.parallel_mode == "auto"


def test_parse_args_respects_overrides(tmp_path) -> None:
    args = _run_parse_args(
        [
            "prog",
            "-p",
            str(tmp_path / "p.csv"),
            "-g",
            str(tmp_path / "g.csv"),
            "--keep-monomorphic",
            "--outputs",
            "manhattan",
            "qq",
            "--methods",
            "GLM",
            "--format",
            "csv",
            "--n-pcs",
            "5",
            "--outputdir",
            str(tmp_path / "results"),
            "--ncpus",
            "4",
            "--parallel-mode",
            "off",
            "--bayesloco-loco-mode",
            "refine",
            "--bayesloco-calibration",
            "gc",
        ]
    )
    assert args.drop_monomorphic is False
    assert args.outputs == ["manhattan", "qq"]
    assert args.methods == "GLM"
    assert args.format == "csv"
    assert args.n_pcs == 5
    assert args.outputdir.endswith("results")
    assert args.ncpus == 4
    assert args.parallel_mode == "off"
    assert args.bayesloco_loco_mode == "refine"
    assert args.bayesloco_calibration == "gc"
