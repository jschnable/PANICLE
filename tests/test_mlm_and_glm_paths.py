import numpy as np
import pandas as pd
import pytest

from panicle.association import glm_fwl_qr, mlm, mlm_loco
from panicle.association.glm import PANICLE_GLM, PANICLE_GLM_MULTI
from panicle.association.mlm import PANICLE_MLM, estimate_variance_components_brent, compute_fast_pvalues
from panicle.association.mlm_loco import PANICLE_MLM_LOCO, PANICLE_MLM_LOCO_MULTI
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.utils.data_types import GenotypeMatrix, KinshipMatrix


def _make_basic_inputs(n_individuals: int = 6, n_markers: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    geno = rng.integers(0, 3, size=(n_individuals, n_markers), dtype=np.int8)
    geno[0, 0] = -9  # inject missing sentinel
    phe = np.column_stack([np.arange(n_individuals), rng.normal(size=n_individuals)])
    kinship = np.eye(n_individuals)
    return geno, phe, kinship


def test_mlm_handles_numpy_missing_and_shapes() -> None:
    geno, phe, kinship = _make_basic_inputs()

    res = PANICLE_MLM(phe, geno, K=kinship, maxLine=2, verbose=False)

    assert res.effects.shape == (geno.shape[1],)
    assert res.se.shape == (geno.shape[1],)
    assert res.pvalues.shape == (geno.shape[1],)
    assert np.all(np.isfinite(res.pvalues))


def test_mlm_accepts_preimputed_genotype_matrix() -> None:
    geno_array, phe, kinship = _make_basic_inputs()
    geno_array[geno_array == -9] = 0  # ensure truly imputed
    genotype = GenotypeMatrix(geno_array, is_imputed=True)

    res = PANICLE_MLM(phe, genotype, K=kinship, maxLine=3, verbose=False)

    assert genotype.is_imputed is True
    assert res.effects.shape[0] == geno_array.shape[1]
    assert np.all(np.isfinite(res.se))


def test_mlm_accepts_1d_phenotype_vector() -> None:
    geno, phe, kinship = _make_basic_inputs()
    trait = phe[:, 1].astype(np.float64)

    res = PANICLE_MLM(trait, geno, K=kinship, maxLine=2, verbose=False)

    assert res.effects.shape == (geno.shape[1],)
    assert np.all(np.isfinite(res.pvalues))


def test_mlm_numpy_missing_matches_genotype_matrix_missing() -> None:
    rng = np.random.default_rng(11)
    n, m = 16, 12
    geno = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    geno[rng.random((n, m)) < 0.15] = -9
    phe = np.column_stack([np.arange(n), rng.normal(size=n)])
    kinship = np.eye(n, dtype=np.float64)

    res_np = PANICLE_MLM(phe, geno, K=kinship, maxLine=4, verbose=False)
    res_gm = PANICLE_MLM(phe, GenotypeMatrix(geno), K=kinship, maxLine=4, verbose=False)

    np.testing.assert_allclose(res_np.effects, res_gm.effects, rtol=1e-8, atol=1e-8, equal_nan=True)
    np.testing.assert_allclose(res_np.se, res_gm.se, rtol=1e-8, atol=1e-8, equal_nan=True)
    np.testing.assert_allclose(res_np.pvalues, res_gm.pvalues, rtol=1e-8, atol=1e-8, equal_nan=True)


def test_mlm_variance_components_brent_produces_positive_components() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.ones((4, 1))
    eigenvals = np.ones(4)

    delta_hat, vg_hat, ve_hat = estimate_variance_components_brent(y, X, eigenvals, verbose=False)

    assert 0 < delta_hat < 1e6
    assert vg_hat > 0
    assert ve_hat >= 0


def test_compute_fast_pvalues_handles_invalid_entries() -> None:
    t_stats = np.array([2.0, np.nan, 0.0])
    dfs = np.array([10.0, 5.0, -1.0])

    pvals = compute_fast_pvalues(t_stats, dfs)

    assert pvals[0] < 0.1
    assert pvals[1] == 1.0
    assert pvals[2] == 1.0


def test_mlm_loco_parallel_path_uses_stubbed_joblib(monkeypatch) -> None:
    rng = np.random.default_rng(2)
    geno = rng.integers(0, 3, size=(5, 4), dtype=np.int8)
    geno[1, 2] = -9  # trigger _subset_genotypes missing handling
    map_df = pd.DataFrame(
        {
            "SNP": [f"s{i}" for i in range(geno.shape[1])],
            "CHROM": ["1", "1", "2", "2"],
            "POS": [10, 20, 30, 40],
        }
    )
    phe = np.column_stack([np.arange(geno.shape[0]), rng.normal(size=geno.shape[0])])

    geno_matrix = GenotypeMatrix(geno)
    loco = PANICLE_K_VanRaden_LOCO(geno_matrix, map_df, maxLine=2, verbose=False)

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return wrapper

    def fake_parallel(n_jobs=None, backend=None):
        def runner(tasks):
            return [task() for task in tasks]
        return runner

    monkeypatch.setattr(mlm_loco, "HAS_JOBLIB", True)
    monkeypatch.setattr(mlm_loco, "Parallel", fake_parallel)
    monkeypatch.setattr(mlm_loco, "delayed", fake_delayed)

    res = PANICLE_MLM_LOCO(
        phe=phe,
        geno=geno_matrix,
        map_data=map_df,
        loco_kinship=loco,
        cpu=2,
        maxLine=2,
        verbose=False,
    )

    assert res.effects.shape == (geno.shape[1],)
    assert res.se.shape == (geno.shape[1],)
    assert res.pvalues.shape == (geno.shape[1],)


def test_mlm_loco_lrt_refinement_uses_prebuilt_fast_path(monkeypatch) -> None:
    rng = np.random.default_rng(3)
    n, m = 12, 8
    geno = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    map_df = pd.DataFrame(
        {
            "SNP": [f"s{i}" for i in range(m)],
            "CHROM": ["1"] * (m // 2) + ["2"] * (m - m // 2),
            "POS": np.arange(1, m + 1),
        }
    )
    phe = np.column_stack([np.arange(n), 0.4 * geno[:, 0].astype(np.float64) + rng.normal(scale=0.2, size=n)])

    geno_matrix = GenotypeMatrix(geno)
    loco = PANICLE_K_VanRaden_LOCO(geno_matrix, map_df, maxLine=4, verbose=False)
    calls = {"batch_calls": 0, "markers_seen": 0}

    def fake_fit_markers_lrt_batch_prebuilt(_y, _x, g_batch, *_args, **_kwargs):
        calls["batch_calls"] += 1
        calls["markers_seen"] += int(g_batch.shape[1])
        n = int(g_batch.shape[1])
        return (
            np.full(n, 0.5, dtype=np.float64),
            np.full(n, 0.1, dtype=np.float64),
            np.full(n, 1.0, dtype=np.float64),
        )

    monkeypatch.setattr(
        mlm_loco,
        "fit_markers_lrt_batch_prebuilt",
        fake_fit_markers_lrt_batch_prebuilt,
    )

    res = PANICLE_MLM_LOCO(
        phe=phe,
        geno=geno_matrix,
        map_data=map_df,
        loco_kinship=loco,
        cpu=2,
        maxLine=4,
        lrt_refinement=True,
        screen_threshold=2.0,  # force all markers into LRT refinement
        lrt_batch_size=2,
        verbose=False,
    )

    assert calls["batch_calls"] >= 1
    assert calls["markers_seen"] >= m
    assert res.effects.shape == (m,)
    assert res.se.shape == (m,)
    assert res.pvalues.shape == (m,)
    assert np.all(np.isfinite(res.pvalues))


def test_mlm_loco_numpy_missing_matches_genotype_matrix_missing() -> None:
    rng = np.random.default_rng(21)
    n, m = 12, 10
    geno = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    geno[rng.random((n, m)) < 0.10] = -9
    map_df = pd.DataFrame(
        {
            "SNP": [f"s{i}" for i in range(m)],
            "CHROM": ["1"] * (m // 2) + ["2"] * (m - m // 2),
            "POS": np.arange(1, m + 1),
        }
    )
    phe = np.column_stack([np.arange(n), rng.normal(size=n)])

    res_np = PANICLE_MLM_LOCO(phe, geno, map_data=map_df, maxLine=4, cpu=1, lrt_refinement=False, verbose=False)
    res_gm = PANICLE_MLM_LOCO(
        phe,
        GenotypeMatrix(geno),
        map_data=map_df,
        maxLine=4,
        cpu=1,
        lrt_refinement=False,
        verbose=False,
    )

    np.testing.assert_allclose(res_np.effects, res_gm.effects, rtol=1e-8, atol=1e-8, equal_nan=True)
    np.testing.assert_allclose(res_np.se, res_gm.se, rtol=1e-8, atol=1e-8, equal_nan=True)
    np.testing.assert_allclose(res_np.pvalues, res_gm.pvalues, rtol=1e-8, atol=1e-8, equal_nan=True)


def test_mlm_loco_multi_matches_single_trait_runs() -> None:
    rng = np.random.default_rng(101)
    n, m = 14, 10
    geno = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    map_df = pd.DataFrame(
        {
            "SNP": [f"s{i}" for i in range(m)],
            "CHROM": ["1"] * (m // 2) + ["2"] * (m - m // 2),
            "POS": np.arange(1, m + 1),
        }
    )

    trait1 = rng.normal(size=n)
    trait2 = 0.2 * geno[:, 0].astype(np.float64) + rng.normal(scale=0.5, size=n)
    phe_multi = np.column_stack([trait1, trait2]).astype(np.float64)

    geno_matrix = GenotypeMatrix(geno)
    loco = PANICLE_K_VanRaden_LOCO(geno_matrix, map_df, maxLine=4, verbose=False)
    trait_names = ["Trait1", "Trait2"]
    multi_results = PANICLE_MLM_LOCO_MULTI(
        phe=phe_multi,
        geno=geno_matrix,
        map_data=map_df,
        trait_names=trait_names,
        loco_kinship=loco,
        maxLine=4,
        cpu=1,
        lrt_refinement=False,
        verbose=False,
    )

    for trait_idx, trait_name in enumerate(trait_names):
        phe_single = np.column_stack([np.arange(n), phe_multi[:, trait_idx]])
        single = PANICLE_MLM_LOCO(
            phe=phe_single,
            geno=geno_matrix,
            map_data=map_df,
            loco_kinship=loco,
            maxLine=4,
            cpu=1,
            lrt_refinement=False,
            verbose=False,
        )
        np.testing.assert_allclose(
            multi_results[trait_name].effects,
            single.effects,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            multi_results[trait_name].se,
            single.se,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            multi_results[trait_name].pvalues,
            single.pvalues,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )


def test_glm_multi_matches_single_trait_runs() -> None:
    rng = np.random.default_rng(131)
    n, m, t = 24, 30, 3
    geno = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    cv = rng.normal(size=(n, 2))
    y_matrix = rng.normal(size=(n, t))
    trait_names = [f"Trait{i+1}" for i in range(t)]

    multi = PANICLE_GLM_MULTI(
        phe=y_matrix,
        geno=geno,
        trait_names=trait_names,
        CV=cv,
        maxLine=8,
        verbose=False,
    )

    for idx, trait_name in enumerate(trait_names):
        phe_single = np.column_stack([np.arange(n), y_matrix[:, idx]])
        single = PANICLE_GLM(
            phe=phe_single,
            geno=geno,
            CV=cv,
            maxLine=8,
            verbose=False,
        )
        np.testing.assert_allclose(
            multi[trait_name].effects,
            single.effects,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            multi[trait_name].se,
            single.se,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            multi[trait_name].pvalues,
            single.pvalues,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )


def test_mlm_errors_on_invalid_inputs() -> None:
    geno, phe, kinship = _make_basic_inputs()
    with pytest.raises(ValueError, match="Phenotype matrix must have 2 columns"):
        PANICLE_MLM(phe[:, :1], geno, K=kinship, verbose=False)

    with pytest.raises(ValueError, match="Covariate matrix must have same number"):
        PANICLE_MLM(phe, geno, K=kinship, CV=np.ones((1, 1)), verbose=False)

    with pytest.raises(ValueError, match="Kinship matrix K is required"):
        PANICLE_MLM(phe, geno, K=None, verbose=False)

    bad_kin = np.eye(phe.shape[0] + 1)
    with pytest.raises(ValueError, match="dimensions must match"):
        PANICLE_MLM(phe, geno, K=bad_kin, verbose=False)


def test_mlm_loco_errors_on_missing_trait_values() -> None:
    geno, phe, _ = _make_basic_inputs()
    map_df = pd.DataFrame(
        {
            "SNP": [f"s{i}" for i in range(geno.shape[1])],
            "CHROM": ["1", "1", "2", "2", "3"][:geno.shape[1]],
            "POS": np.arange(1, geno.shape[1] + 1),
        }
    )
    phe[1, 1] = np.nan

    with pytest.raises(ValueError, match="missing/non-finite"):
        PANICLE_MLM_LOCO(phe, geno, map_data=map_df, verbose=False)


def test_mlm_uses_provided_eigen_and_kinship_matrix_and_cpu_zero() -> None:
    geno, phe, _ = _make_basic_inputs(n_individuals=5, n_markers=3, seed=3)
    kin_np = np.eye(geno.shape[0])
    eigenvals, eigenvecs = np.linalg.eigh(kin_np)
    eigenK = {"eigenvals": eigenvals, "eigenvecs": eigenvecs.astype(np.float32)}
    kinship_obj = KinshipMatrix(kin_np)

    res = PANICLE_MLM(phe, geno, K=kinship_obj, eigenK=eigenK, cpu=0, maxLine=2, verbose=False)

    assert res.effects.shape == (geno.shape[1],)
    assert np.all(np.isfinite(res.se))


def test_glm_impute_numpy_major_and_fill_override() -> None:
    batch = np.array(
        [
            [0.0, -9.0, 3.0],
            [2.0, np.nan, -9.0],
        ]
    )

    imputed_major = glm_fwl_qr._impute_numpy_batch_major_allele(batch)
    assert imputed_major[0, 1] == 0.0  # defaults to 0 when no valid values
    assert imputed_major[1, 2] == 3.0  # unexpected genotype fallback uses observed value

    imputed_fill = glm_fwl_qr._impute_numpy_batch_major_allele(batch, fill_value=1.5)
    assert np.all(imputed_fill[:, 1] == 1.5)
    assert imputed_fill[1, 2] == 1.5  # only the missing element gets the fill value


def test_glm_ultrafast_runs_with_covariates_and_genotype_matrix() -> None:
    geno = np.array(
        [
            [0, 1],
            [1, 0],
            [2, 1],
            [1, 2],
        ],
        dtype=np.int8,
    )
    genotype = GenotypeMatrix(geno, is_imputed=True)
    phe = np.column_stack([np.arange(geno.shape[0]), np.array([0.5, 1.0, 1.5, 2.0])])
    covariate = np.array([[0.0], [1.0], [0.0], [1.0]])

    res = glm_fwl_qr.PANICLE_GLM_ultrafast(
        phe=phe,
        geno=genotype,
        CV=covariate,
        maxLine=1,
        verbose=False,
        missing_fill_value=0.0,
    )

    assert res.effects.shape == (geno.shape[1],)
    assert res.se.shape == (geno.shape[1],)
    assert res.pvalues.shape == (geno.shape[1],)
    assert np.all(np.isfinite(res.pvalues))


def test_glm_ultrafast_rejects_bad_phenotype_shape() -> None:
    phe = np.array([[1.0, 2.0, 3.0]])
    geno = np.ones((1, 1), dtype=np.int8)

    with pytest.raises(ValueError):
        glm_fwl_qr.PANICLE_GLM_ultrafast(phe=phe, geno=geno, verbose=False)


def test_glm_ultrafast_handles_singular_covariates_and_df_guard() -> None:
    geno = np.array([[0, 1], [1, 0]], dtype=np.int8)
    phe = np.column_stack([np.arange(2), np.array([0.5, 1.0])])
    cov = np.ones((2, 1), dtype=np.float32)  # perfect collinearity with intercept

    with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
        glm_fwl_qr.PANICLE_GLM_ultrafast(phe=phe, geno=geno, CV=cov, verbose=False)

    phe_long = np.column_stack([np.arange(4), np.array([0.1, 0.2, 0.3, 0.4])])
    cov_long = np.ones((4, 1), dtype=np.float32)
    geno_long = np.array([[0, 1], [1, 0], [0, 0], [2, 1]], dtype=np.int8)
    res = glm_fwl_qr.PANICLE_GLM_ultrafast(phe=phe_long, geno=geno_long, CV=cov_long, verbose=False)
    assert res.effects.shape[0] == geno.shape[1]
