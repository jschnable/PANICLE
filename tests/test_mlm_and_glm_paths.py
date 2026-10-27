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


def test_mlm_loco_multichrom_cpu_invariant() -> None:
    # Chromosomes run sequentially; parallelism is marker-level inside
    # PANICLE_MLM (pinned to `cpu`). Results must be invariant to the CPU
    # budget, so cpu=1 and cpu=2 must produce identical output on a
    # multi-chromosome map.
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

    kwargs = dict(phe=phe, geno=geno_matrix, map_data=map_df,
                  loco_kinship=loco, maxLine=2, verbose=False)
    res1 = PANICLE_MLM_LOCO(cpu=1, **kwargs)
    res2 = PANICLE_MLM_LOCO(cpu=2, **kwargs)

    assert res1.effects.shape == (geno.shape[1],)
    assert res1.se.shape == (geno.shape[1],)
    assert res1.pvalues.shape == (geno.shape[1],)
    np.testing.assert_array_equal(res1.pvalues, res2.pvalues)
    np.testing.assert_array_equal(res1.effects, res2.effects)
    np.testing.assert_array_equal(res1.se, res2.se)


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
        keep="all",
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


def test_mlm_loco_multi_cpu0_uses_affinity_aware_count(monkeypatch) -> None:
    # cpu=0 ("all cores") must expand via the affinity/cgroup-aware helper, not
    # multiprocessing.cpu_count(), so PANICLE never oversubscribes on a
    # cgroup/SLURM-limited node.
    rng = np.random.default_rng(202)
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

    calls = {"n": 0}

    def fake_available_cpu_count() -> int:
        calls["n"] += 1
        return 1

    monkeypatch.setattr(mlm_loco, "available_cpu_count", fake_available_cpu_count)

    multi_results = PANICLE_MLM_LOCO_MULTI(
        phe=phe_multi,
        geno=geno_matrix,
        map_data=map_df,
        trait_names=["Trait1", "Trait2"],
        loco_kinship=loco,
        maxLine=4,
        cpu=0,
        lrt_refinement=False,
        verbose=False,
    )

    assert calls["n"] >= 1, "cpu=0 MULTI path must expand via available_cpu_count()"
    assert set(multi_results) == {"Trait1", "Trait2"}


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


def test_glm_prefetch_preserves_single_and_joint_results(monkeypatch):
    rng = np.random.default_rng(913)
    raw = rng.integers(0, 3, size=(80, 137), dtype=np.int8)
    raw.flags.writeable = False
    geno = GenotypeMatrix(raw, is_imputed=True, precompute_alleles=False)
    y = rng.normal(size=(80, 3))
    cv = rng.normal(size=(80, 2))
    phe = np.column_stack([np.arange(80), y[:, 0]])
    # More than two batches, including a partial final batch.
    monkeypatch.setenv("PANICLE_GLM_PREFETCH", "off")
    single = PANICLE_GLM(phe, geno, CV=cv, maxLine=19, cpu=1, verbose=False)
    joint = PANICLE_GLM_MULTI(y, geno, CV=cv, maxLine=19, cpu=1, verbose=False)
    monkeypatch.delenv("PANICLE_GLM_PREFETCH")
    prefetched = PANICLE_GLM(phe, geno, CV=cv, maxLine=19, cpu=4, verbose=False)
    joint_prefetched = PANICLE_GLM_MULTI(y, geno, CV=cv, maxLine=19, cpu=4, verbose=False)
    np.testing.assert_array_equal(single.to_numpy(), prefetched.to_numpy())
    for name in joint:
        np.testing.assert_array_equal(joint[name].to_numpy(), joint_prefetched[name].to_numpy())


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
    phe = phe.astype(object)
    phe[:, 0] = np.array([f"sample_{i}" for i in range(phe.shape[0])], dtype=object)
    phe[1, 1] = np.nan

    with pytest.raises(ValueError, match="sample_1"):
        PANICLE_MLM_LOCO(phe, geno, map_data=map_df, verbose=False)


def test_mlm_reports_missing_phenotype_sample_ids() -> None:
    geno, phe, kinship = _make_basic_inputs()
    phe = phe.astype(object)
    phe[:, 0] = np.array([f"line_{i}" for i in range(phe.shape[0])], dtype=object)
    phe[2, 1] = np.inf

    with pytest.raises(ValueError, match="line_2"):
        PANICLE_MLM(phe, geno, K=kinship, verbose=False)


def test_glm_ultrafast_reports_missing_phenotype_sample_ids() -> None:
    geno, phe, _ = _make_basic_inputs()
    phe = phe.astype(object)
    phe[:, 0] = np.array([f"line_{i}" for i in range(phe.shape[0])], dtype=object)
    phe[3, 1] = np.nan

    with pytest.raises(ValueError, match="line_3"):
        glm_fwl_qr.PANICLE_GLM_ultrafast(phe, geno, verbose=False)


def test_mlm_loco_errors_on_mismatched_loco_kinship_dimensions() -> None:
    rng = np.random.default_rng(17)
    geno = rng.integers(0, 3, size=(8, 6), dtype=np.int8)
    map_df = pd.DataFrame(
        {
            "SNP": [f"s{i}" for i in range(geno.shape[1])],
            "CHROM": ["1", "1", "1", "2", "2", "2"],
            "POS": np.arange(1, geno.shape[1] + 1),
        }
    )
    phe = np.column_stack([np.arange(6), rng.normal(size=6)])

    full_loco = PANICLE_K_VanRaden_LOCO(geno, map_df, verbose=False)

    with pytest.raises(ValueError, match="LOCO kinship dimensions must match"):
        PANICLE_MLM_LOCO(
            phe=phe,
            geno=geno[:6, :],
            map_data=map_df,
            loco_kinship=full_loco,
            verbose=False,
        )


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


def test_brent_batch_matches_scalar_variance_components() -> None:
    from panicle.association.mlm import (
        estimate_variance_components_brent,
        estimate_variance_components_brent_batch,
    )

    rng = np.random.default_rng(5)
    n, q0, T = 60, 2, 12
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eig = np.sort(rng.gamma(2.0, 0.5, size=n))[::-1].copy()
    eig[-1] = 0.0  # clamped near-zero eigenvalue like a real LOCO kinship
    X = U.T @ np.column_stack([np.ones(n), rng.normal(size=(n, q0 - 1))])
    g = U @ (np.sqrt(np.maximum(eig, 1e-6))[:, None] * rng.normal(size=(n, T)))
    h2 = np.linspace(0.02, 0.95, T)
    Y = U.T @ (np.sqrt(h2) * g / g.std(axis=0) + np.sqrt(1 - h2) * rng.normal(size=(n, T)))

    delta, vg, ve = estimate_variance_components_brent_batch(Y, X, eig)
    ref = np.array([estimate_variance_components_brent(Y[:, t], X, eig) for t in range(T)])
    np.testing.assert_allclose(delta, ref[:, 0], rtol=1e-6)
    np.testing.assert_allclose(vg, ref[:, 1], rtol=1e-6)
    np.testing.assert_allclose(ve, ref[:, 2], rtol=1e-6)


def test_wald_trait_block_matches_per_trait_kernel() -> None:
    rng = np.random.default_rng(6)
    n, m, q0, T = 80, 300, 2, 7
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eig = np.sort(rng.gamma(2.0, 0.5, size=n))[::-1].copy()
    G = rng.integers(0, 3, size=(n, m)).astype(np.float32)
    G_eig = (U.T.astype(np.float32) @ G).astype(np.float32)
    X_eig = U.T @ np.column_stack([np.ones(n), rng.normal(size=(n, q0 - 1))])
    Y = rng.normal(size=(n, T))
    Y[:, 0] += 0.8 * G[:, 3]
    Y_eig = U.T @ Y

    eff_b, se_b, p_b = mlm_loco._run_wald_trait_block(
        Y_eig, X_eig, eig, G_eig, vc_method="BRENT", maxLine=128, cpu=1
    )
    for t in range(T):
        eff, se, p = mlm_loco._run_wald_with_pretransformed_genotypes(
            Y_eig[:, t], X_eig, eig, G_eig, vc_method="BRENT", maxLine=128, cpu=1
        )
        # Same float32 cross-products, different BLAS reduction order.
        np.testing.assert_allclose(eff_b[t], eff, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(se_b[t], se, rtol=1e-4)
        np.testing.assert_allclose(-np.log10(p_b[t]), -np.log10(p), atol=1e-3)
    assert p_b[0].argmin() == 3


def test_mlm_loco_multi_trait_block_size_invariant() -> None:
    rng = np.random.default_rng(8)
    n, m, T = 40, 60, 9
    geno = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    map_df = pd.DataFrame(
        {
            "SNP": [f"s{i}" for i in range(m)],
            "CHROM": ["1"] * (m // 2) + ["2"] * (m - m // 2),
            "POS": np.arange(1, m + 1),
        }
    )
    phe = rng.normal(size=(n, T))
    geno_matrix = GenotypeMatrix(geno)
    loco = PANICLE_K_VanRaden_LOCO(geno_matrix, map_df, maxLine=16, verbose=False)
    kwargs = dict(
        phe=phe, geno=geno_matrix, map_data=map_df, loco_kinship=loco,
        maxLine=16, cpu=1, lrt_refinement=False, verbose=False, keep="all",
    )
    whole = PANICLE_MLM_LOCO_MULTI(trait_block_size=T, **kwargs)
    split = PANICLE_MLM_LOCO_MULTI(trait_block_size=4, **kwargs)
    for name in whole:
        np.testing.assert_allclose(
            -np.log10(whole[name].pvalues), -np.log10(split[name].pvalues), atol=1e-3
        )
    with pytest.raises(ValueError):
        PANICLE_MLM_LOCO_MULTI(trait_block_size=0, **kwargs)


def _multi_trait_fixture(seed: int = 11, n: int = 36, m: int = 48, T: int = 6):
    rng = np.random.default_rng(seed)
    geno = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    map_df = pd.DataFrame(
        {
            "SNP": [f"s{i}" for i in range(m)],
            "CHROM": ["1"] * (m // 2) + ["2"] * (m - m // 2),
            "POS": np.arange(1, m + 1),
        }
    )
    phe = rng.normal(size=(n, T))
    phe[:, 0] += 1.5 * geno[:, 3]
    phe[:, 2] += 1.5 * geno[:, 30]
    geno_matrix = GenotypeMatrix(geno)
    loco = PANICLE_K_VanRaden_LOCO(geno_matrix, map_df, maxLine=16, verbose=False)
    names = [f"gene{t}" for t in range(T)]
    kwargs = dict(
        phe=phe, geno=geno_matrix, map_data=map_df, trait_names=names, loco_kinship=loco,
        maxLine=16, cpu=1, verbose=False,
    )
    return names, map_df, kwargs


def test_estimate_null_ml_batch_matches_scalar() -> None:
    from panicle.association.mlm import (
        _calculate_neg_ml_likelihood,
        estimate_null_ml_batch,
        estimate_variance_components_brent,
    )

    rng = np.random.default_rng(9)
    n, q0, T = 50, 2, 8
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eig = np.sort(rng.gamma(2.0, 0.5, size=n))[::-1].copy()
    X = U.T @ np.column_stack([np.ones(n), rng.normal(size=(n, q0 - 1))])
    Y = U.T @ rng.normal(size=(n, T))
    h2_b, nll_b = estimate_null_ml_batch(Y, X, eig)
    for t in range(T):
        _, vg, ve = estimate_variance_components_brent(Y[:, t], X, eig, use_ml=True)
        h2 = vg / (vg + ve)
        np.testing.assert_allclose(h2_b[t], h2, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(nll_b[t], _calculate_neg_ml_likelihood(h2, Y[:, t], X, eig), rtol=1e-8)


@pytest.mark.parametrize("lrt_refinement", [False, True])
def test_mlm_loco_multi_hits_store_matches_full_results(lrt_refinement: bool) -> None:
    from panicle.association.multi_trait_store import MultiTraitResultStore

    names, map_df, kwargs = _multi_trait_fixture()
    thr = 1e-3
    full = PANICLE_MLM_LOCO_MULTI(keep="all", lrt_refinement=lrt_refinement, **kwargs)
    store = PANICLE_MLM_LOCO_MULTI(keep="hits", p_threshold=thr, lrt_refinement=lrt_refinement, **kwargs)
    assert isinstance(store, MultiTraitResultStore)
    assert store.keep == "hits" and store.complete and not store.dense
    with pytest.raises(RuntimeError):
        store["gene0"]

    hits = store.hits
    expected_rows = 0
    for name in names:
        res = full[name]
        idx = np.nonzero(res.pvalues <= thr)[0]
        expected_rows += idx.size
        th = store.trait_hits(name)
        assert th.shape[0] == idx.size
        got = th.set_index("MARKER")
        for i in idx:
            marker = map_df.loc[i, "SNP"]
            np.testing.assert_allclose(got.loc[marker, "P_value"], res.pvalues[i], rtol=1e-12)
            np.testing.assert_allclose(got.loc[marker, "Effect"], res.effects[i], rtol=1e-12)
            np.testing.assert_allclose(got.loc[marker, "SE"], res.se[i], rtol=1e-12)
        # top marker per trait
        row = store.trait_summary.set_index("Trait").loc[name]
        assert row["top_MARKER"] == map_df.loc[int(res.pvalues.argmin()), "SNP"]
        np.testing.assert_allclose(row["top_P_value"], res.pvalues.min(), rtol=1e-12)
    assert hits.shape[0] == expected_rows
    assert expected_rows >= 2  # the two planted signals
    assert set(hits["Trait"]) >= {"gene0", "gene2"}
    assert set(hits.columns) == {"Trait", "MARKER", "CHROM", "POS", "Effect", "SE", "P_value"}


def test_mlm_loco_multi_dense_store_streams_and_reopens(tmp_path) -> None:
    from panicle.association.multi_trait_store import MultiTraitResultStore

    names, map_df, kwargs = _multi_trait_fixture()
    full = PANICLE_MLM_LOCO_MULTI(keep="all", lrt_refinement=False, **kwargs)
    out = tmp_path / "eqtl_store"
    store = PANICLE_MLM_LOCO_MULTI(
        keep="all", output_dir=out, p_threshold=1e-3, lrt_refinement=False, dense_dtype=np.float64, **kwargs
    )
    assert store.dense and store.complete
    assert store.pvalues.shape == (len(names), map_df.shape[0])
    for name in names:
        np.testing.assert_allclose(store[name].pvalues, full[name].pvalues, rtol=1e-12)
        np.testing.assert_allclose(store[name].effects, full[name].effects, rtol=1e-12)
        np.testing.assert_allclose(store[name].se, full[name].se, rtol=1e-12)
    assert list(store.keys()) == names and "gene1" in store and len(store) == len(names)

    for fname in ("effects.npy", "se.npy", "pvalues.npy", "hits.csv", "trait_summary.csv", "metadata.json"):
        assert (out / fname).exists()
    hits_on_disk = pd.read_csv(out / "hits.csv")
    assert hits_on_disk.shape[0] == store.hits.shape[0]

    reopened = MultiTraitResultStore.open(out)
    assert reopened.keep == "all" and reopened.trait_names == names
    np.testing.assert_allclose(reopened["gene2"].pvalues, full["gene2"].pvalues, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(reopened.pvalues), np.asarray(store.pvalues))
    assert reopened.hits.shape == store.hits.shape
    pd.testing.assert_frame_equal(
        reopened.trait_summary.reset_index(drop=True), pd.read_csv(out / "trait_summary.csv")
    )

    # float32 dense storage keeps p-values in float64
    store32 = PANICLE_MLM_LOCO_MULTI(keep="all", output_dir=tmp_path / "f32", lrt_refinement=False, **kwargs)
    assert store32.effects.dtype == np.float32 and store32.pvalues.dtype == np.float64
    np.testing.assert_allclose(store32["gene0"].effects, full["gene0"].effects, rtol=1e-5)


def test_mlm_loco_multi_rejects_bad_keep() -> None:
    _, _, kwargs = _multi_trait_fixture()
    with pytest.raises(ValueError):
        PANICLE_MLM_LOCO_MULTI(keep="some", **kwargs)
