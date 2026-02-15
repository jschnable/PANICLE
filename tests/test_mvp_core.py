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

    # Results are nested by trait name (single trait uses "Trait")
    assert res["results"]["Trait"]["GLM"] is dummy
    assert res["summary"]["significant_markers"]["Trait"]["GLM"] == 1
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

    # Results are nested by trait name
    assert res["results"]["Trait"]["FarmCPUResampling"] is dummy_resampling
    assert res["summary"]["significant_markers"]["Trait"]["FarmCPUResampling"] >= 0
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

    # Results are now nested by trait name
    results = {
        "data": {"map": geno_map},
        "results": {"Trait": {"GLM": dummy_result}},
        "summary": {
            "methods_run": ["GLM"],
            "total_individuals": genotype.n_individuals,
            "total_markers": genotype.n_markers,
            "n_traits": 1,
            "trait_names": ["Trait"],
            "significant_markers": {"Trait": {"GLM": 1}},
            "runtime": {"GLM_Trait": 0.1, "total": 0.2},
        },
        "files": [],
    }

    files = mvp.save_results_to_files(results, str(tmp_path / "out"), verbose=False)

    assert any(f.endswith("_summary.txt") for f in files)
    assert any("GLM_results.csv" in f for f in files)
    for f in files:
        assert Path(f).exists()


def test_panicle_auto_matches_ids_and_subsets_genotype(monkeypatch, tmp_path) -> None:
    geno_file = tmp_path / "geno.csv"
    pd.DataFrame(
        {
            "ID": ["A", "B", "C"],
            "m1": [0, 1, 2],
            "m2": [2, 1, 0],
        }
    ).to_csv(geno_file, index=False)
    geno_map = GenotypeMap(
        pd.DataFrame({"SNP": ["m1", "m2"], "CHROM": [1, 1], "POS": [1, 2]})
    )
    phe_file = tmp_path / "phe.csv"
    pd.DataFrame({"ID": ["B", "D", "A"], "Trait": [1.0, 2.0, 3.0]}).to_csv(phe_file, index=False)

    captured = {}
    dummy = DummyAssocResult(2, np.array([0.2, 0.3]))

    def fake_glm(**kwargs):
        captured["phe"] = kwargs["phe"]
        captured["geno"] = kwargs["geno"].get_batch(0, kwargs["geno"].n_markers)
        return dummy

    monkeypatch.setattr(mvp, "PANICLE_GLM", fake_glm)
    monkeypatch.setattr(mvp, "PANICLE_Report", lambda **kwargs: {"files_created": []})

    res = mvp.PANICLE(
        phe=str(phe_file),
        geno=str(geno_file),
        map_data=geno_map,
        method=["GLM"],
        file_output=False,
        verbose=False,
    )

    assert captured["phe"][:, 0].tolist() == ["B", "A"]
    np.testing.assert_array_equal(captured["geno"], np.array([[1, 1], [0, 2]], dtype=np.int8))
    assert res["summary"]["total_individuals"] == 2
    assert res["summary"]["sample_matching"]["n_common"] == 2
    assert res["summary"]["sample_matching"]["n_phenotype_dropped"] == 1
    assert res["summary"]["sample_matching"]["n_genotype_dropped"] == 1


def test_panicle_excludes_missing_trait_values_per_trait(monkeypatch, tmp_path) -> None:
    geno_file = tmp_path / "geno.csv"
    pd.DataFrame(
        {
            "ID": ["A", "B", "C"],
            "m1": [0, 1, 2],
            "m2": [2, 1, 0],
        }
    ).to_csv(geno_file, index=False)
    geno_map = GenotypeMap(
        pd.DataFrame({"SNP": ["m1", "m2"], "CHROM": [1, 1], "POS": [1, 2]})
    )
    phe_file = tmp_path / "phe_nan.csv"
    pd.DataFrame({"ID": ["A", "B", "C"], "Trait": [1.0, np.nan, 3.0]}).to_csv(phe_file, index=False)

    captured = {}
    dummy = DummyAssocResult(2, np.array([0.4, 0.5]))

    def fake_glm(**kwargs):
        captured["phe"] = kwargs["phe"]
        captured["geno_n"] = kwargs["geno"].n_individuals
        return dummy

    monkeypatch.setattr(mvp, "PANICLE_GLM", fake_glm)
    monkeypatch.setattr(mvp, "PANICLE_Report", lambda **kwargs: {"files_created": []})

    res = mvp.PANICLE(
        phe=str(phe_file),
        geno=str(geno_file),
        map_data=geno_map,
        method=["GLM"],
        file_output=False,
        verbose=False,
    )

    assert captured["geno_n"] == 2
    assert captured["phe"][:, 0].tolist() == ["A", "C"]
    assert np.all(np.isfinite(captured["phe"][:, 1].astype(float)))
    assert res["summary"]["trait_sample_sizes"]["Trait"] == 2
