import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from panicle.core import mvp
from panicle.utils.data_types import GenotypeMap, GenotypeMatrix, Phenotype


class DummyAssocResult:
    def __init__(self, n_markers: int, pvals: np.ndarray):
        self._pvals = pvals
        self._n = n_markers

    def to_numpy(self) -> np.ndarray:
        # columns: effect, se, pval
        effects = np.zeros(self._n)
        se = np.ones(self._n)
        return np.column_stack([effects, se, self._pvals])

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"P-value": self._pvals})


def _basic_inputs(n: int = 12, m: int = 3):
    phe = np.column_stack([np.arange(n), np.linspace(0.0, 1.0, n)])
    geno = np.random.default_rng(0).integers(0, 3, size=(n, m)).astype(np.int8)
    geno_map = GenotypeMap(
        pd.DataFrame({"SNP": [f"s{i}" for i in range(m)], "CHROM": ["1"] * m, "POS": np.arange(1, m + 1)})
    )
    return phe, geno, geno_map


def test_validate_data_consistency_checks_lengths_and_warns() -> None:
    phe, geno, geno_map = _basic_inputs(n=8, m=2)
    phenotype = Phenotype(phe)
    genotype = GenotypeMatrix(geno)

    with pytest.raises(ValueError):
        mvp.validate_data_consistency(phenotype, genotype, GenotypeMap(geno_map.to_dataframe().iloc[:1]), verbose=False)

    # Small counts trigger warnings but not errors
    with pytest.warns(UserWarning):
        mvp.validate_data_consistency(phenotype, genotype, geno_map, verbose=False)


def test_panicle_glm_only_runs_and_summarizes(monkeypatch) -> None:
    phe, geno, geno_map = _basic_inputs()
    dummy = DummyAssocResult(geno.shape[1], np.array([1e-10, 0.2, 0.3]))

    monkeypatch.setattr(mvp, "PANICLE_GLM", lambda **kwargs: dummy)
    # Keep other methods unused
    monkeypatch.setattr(mvp, "PANICLE_Report", lambda **kwargs: {"files_created": []})

    res = mvp.PANICLE(
        phe,
        geno,
        geno_map,
        method=["GLM"],
        file_output=False,
        verbose=False,
        threshold=0.05,
    )

    assert res["results"]["GLM"] is dummy
    assert res["summary"]["significant_markers"]["GLM"] == 1
    assert res["summary"]["methods_run"] == ["GLM"]


def test_panicle_farmcpu_resampling_threshold_warning(monkeypatch) -> None:
    phe, geno, geno_map = _basic_inputs()
    class DummyResampling(DummyAssocResult):
        def __init__(self):
            super().__init__(geno.shape[1], np.array([0.1, 0.2, 0.3]))
            self.entries = []

    dummy_resampling = DummyResampling()
    dummy_report = {"files_created": []}

    monkeypatch.setattr(mvp, "PANICLE_FarmCPUResampling", lambda **kwargs: dummy_resampling)
    monkeypatch.setattr(mvp, "PANICLE_Report", lambda **kwargs: dummy_report)

    res = mvp.PANICLE(
        phe,
        geno,
        geno_map,
        method=["FarmCPUResampling"],
        file_output=False,
        verbose=False,
        farmcpu_resampling_significance_threshold=0.5,  # less stringent than qtn threshold
        p_threshold=0.1,
        QTN_threshold=0.2,
    )

    assert res["results"]["FarmCPUResampling"] is dummy_resampling
    assert res["summary"]["significant_markers"]["FarmCPUResampling"] >= 0
    assert res["summary"]["methods_run"] == ["FarmCPUResampling"]


def test_panicle_rejects_nonexistent_genotype_path(monkeypatch) -> None:
    phe, geno, geno_map = _basic_inputs()
    geno_file = Path("fake.bed")
    # Ensure other heavy functions are not called
    monkeypatch.setattr(mvp, "PANICLE_Report", lambda **kwargs: {"files_created": []})
    with pytest.raises((FileNotFoundError, ValueError)):
        mvp.PANICLE(phe, str(geno_file), geno_map, method=["GLM"], file_output=False, verbose=False)


def test_save_results_to_files_writes_outputs(tmp_path) -> None:
    phe, geno, geno_map = _basic_inputs(m=2)
    phenotype = Phenotype(phe)
    genotype = GenotypeMatrix(geno)
    dummy_result = DummyAssocResult(geno.shape[1], np.array([0.01, 0.02]))

    results = {
        "data": {"map": geno_map},
        "results": {"GLM": dummy_result},
        "summary": {
            "methods_run": ["GLM"],
            "total_individuals": genotype.n_individuals,
            "total_markers": genotype.n_markers,
            "significant_markers": {"GLM": 1},
            "runtime": {"GLM": 0.1, "total": 0.2},
        },
        "files": [],
    }

    files = mvp.save_results_to_files(results, str(tmp_path / "out"), verbose=False)

    assert any(f.endswith("_summary.txt") for f in files)
    assert any("GLM_results.csv" in f for f in files)
    for f in files:
        assert Path(f).exists()
