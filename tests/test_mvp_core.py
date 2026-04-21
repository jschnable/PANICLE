import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from panicle.core import mvp
from panicle.association.mlm_loco import PANICLE_MLM_LOCO
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
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
        min_mac=0,
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


def test_panicle_computes_internal_pcs_and_appends_to_covariates(monkeypatch) -> None:
    phe, geno, geno_map = _basic_inputs(n=6, m=4)
    phenotype = pd.DataFrame({"ID": phe[:, 0], "Trait": phe[:, 1].astype(float)})
    external_covariates = np.arange(12, dtype=float).reshape(6, 2)
    pcs = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 6),
            np.linspace(2.0, -2.0, 6),
        ]
    )
    captured = {}
    dummy = DummyAssocResult(geno.shape[1], np.full(geno.shape[1], 0.25))

    def fake_pca(*, M, pcs_keep, verbose):
        captured["pca_input_n"] = M.n_individuals
        captured["pcs_keep"] = pcs_keep
        return pcs[:, :pcs_keep]

    def fake_glm(**kwargs):
        captured["glm_cv"] = kwargs["CV"]
        return dummy

    monkeypatch.setattr(mvp, "PANICLE_PCA", fake_pca)
    monkeypatch.setattr(mvp, "PANICLE_GLM", fake_glm)
    monkeypatch.setattr(mvp, "PANICLE_Report", lambda **kwargs: {"files_created": []})

    res = mvp.PANICLE(
        phe=phenotype,
        geno=geno,
        map_data=geno_map,
        CV=external_covariates,
        n_pcs=2,
        method=["GLM"],
        file_output=False,
        verbose=False,
    )

    expected_covariates = np.column_stack([external_covariates, pcs])
    np.testing.assert_allclose(captured["glm_cv"], expected_covariates)
    np.testing.assert_allclose(res["data"]["pcs"], pcs)
    np.testing.assert_allclose(res["data"]["covariates"], expected_covariates)
    assert captured["pca_input_n"] == geno.shape[0]
    assert captured["pcs_keep"] == 2


def test_panicle_mlm_matches_direct_loco_when_trait_contains_missing_values() -> None:
    rng = np.random.default_rng(7)
    n_individuals = 18
    n_markers = 24
    ids = np.array([f"L{i:03d}" for i in range(n_individuals)])
    geno = rng.integers(0, 3, size=(n_individuals, n_markers), dtype=np.int8)
    map_df = pd.DataFrame(
        {
            "SNP": [f"s{i}" for i in range(n_markers)],
            "CHROM": ["1"] * (n_markers // 2) + ["2"] * (n_markers - n_markers // 2),
            "POS": np.arange(1, n_markers + 1),
        }
    )
    trait = 0.75 * geno[:, 0].astype(np.float64) + rng.normal(scale=0.3, size=n_individuals)
    trait[[2, 5, 11]] = np.nan
    phenotype_df = pd.DataFrame({"ID": ids, "Trait": trait})

    high_level = mvp.PANICLE(
        phe=phenotype_df,
        geno=GenotypeMatrix(geno),
        map_data=GenotypeMap(map_df),
        method=["MLM"],
        file_output=False,
        lrt_refinement=False,
        verbose=False,
        min_mac=0,
    )

    mask = np.isfinite(trait)
    subset_indices = np.where(mask)[0]
    subset_geno = GenotypeMatrix(geno).subset_individuals(subset_indices, materialize=True)
    subset_phe = np.column_stack([ids[mask], trait[mask]])
    subset_loco = PANICLE_K_VanRaden_LOCO(subset_geno, map_df, verbose=False)
    direct = PANICLE_MLM_LOCO(
        phe=subset_phe,
        geno=subset_geno,
        map_data=map_df,
        loco_kinship=subset_loco,
        lrt_refinement=False,
        verbose=False,
    )

    high_level_res = high_level["results"]["Trait"]["MLM"]
    np.testing.assert_allclose(
        high_level_res.effects,
        direct.effects,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        high_level_res.se,
        direct.se,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        high_level_res.pvalues,
        direct.pvalues,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    assert high_level["summary"]["trait_sample_sizes"]["Trait"] == int(mask.sum())
