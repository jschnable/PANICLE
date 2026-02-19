import numpy as np
import pandas as pd
import pytest

from panicle.association.bayes_loco import PANICLE_BayesLOCO
from panicle.pipelines.gwas import GWASPipeline


def _tiny_inputs(n: int = 24, m: int = 64, seed: int = 123):
    rng = np.random.default_rng(seed)
    ids = np.array([f"id{i:03d}" for i in range(n)])
    geno = rng.integers(0, 3, size=(n, m)).astype(np.int8)
    # Inject a sparse signal.
    causal = np.array([2, 7, 19, 33], dtype=int)
    score = geno[:, causal].sum(axis=1).astype(float)
    y = 0.7 * score + rng.normal(0.0, 1.0, size=n)
    phe = np.column_stack([ids, y])
    cv = np.column_stack([rng.normal(size=n), rng.integers(0, 3, size=n)]).astype(float)
    map_df = pd.DataFrame(
        {
            "SNP": [f"snp{i:05d}" for i in range(m)],
            "CHROM": [f"Chr{(i % 4) + 1:02d}" for i in range(m)],
            "POS": np.arange(1, m + 1),
        }
    )
    return phe, geno, map_df, cv


def _base_cfg():
    return {
        "max_iter": 12,
        "patience": 4,
        "prior_tune_stage1_max_iter": 4,
        "prior_tune_stage2_max_iter": 6,
        "batch_markers_fit": 16,
        "batch_markers_test": 32,
        "prior_tune_pi_grid": (0.02,),
        "prior_tune_slab_scale_grid": (1.0,),
        "screening_warmup_epochs": 2,
        "verification_interval": 2,
        "calibrate_stat_scale": "none",
        "random_seed": 123,
        "deterministic": True,
    }


def test_bayes_loco_basic_api_runs_and_returns_metadata():
    phe, geno, map_df, cv = _tiny_inputs()
    res = PANICLE_BayesLOCO(
        phe=phe,
        geno=geno,
        map_data=map_df,
        CV=cv,
        verbose=False,
        bl_config=_base_cfg(),
    )
    assert res.n_markers == geno.shape[1]
    assert np.all(np.isfinite(res.effects))
    assert np.all(np.isfinite(res.pvalues))
    assert np.min(res.pvalues) >= 0.0
    assert np.max(res.pvalues) <= 1.0
    assert isinstance(res.metadata, dict)
    for key in [
        "method",
        "h2_hat",
        "prior_pi_selected",
        "sigma_slab2_selected",
        "sigma_spike2_effective",
        "timing_total_s",
        "pass_equiv_total",
    ]:
        assert key in res.metadata
    assert res.metadata["method"] == "BAYESLOCO"
    assert res.metadata["sigma_spike2_effective"] <= res.metadata["sigma_slab2_selected"] * 0.1 + 1e-20


def test_bayes_loco_rejects_binary_trait_v1():
    phe, geno, map_df, cv = _tiny_inputs()
    phe[:, 1] = (np.arange(phe.shape[0]) % 2).astype(float)
    with pytest.raises(NotImplementedError):
        PANICLE_BayesLOCO(
            phe=phe,
            geno=geno,
            map_data=map_df,
            CV=cv,
            verbose=False,
        )


def test_pipeline_runs_bayesloco(tmp_path):
    phe, geno, map_df, _ = _tiny_inputs(n=20, m=40, seed=999)
    pheno_df = pd.DataFrame({"ID": phe[:, 0], "Trait": phe[:, 1].astype(float)})
    geno_df = pd.DataFrame(geno, columns=map_df["SNP"])
    geno_df.insert(0, "ID", phe[:, 0])

    phe_file = tmp_path / "phe.csv"
    geno_file = tmp_path / "geno.csv"
    map_file = tmp_path / "map.csv"
    pheno_df.to_csv(phe_file, index=False)
    geno_df.to_csv(geno_file, index=False)
    map_df.to_csv(map_file, index=False)

    output_dir = tmp_path / "out"
    pipeline = GWASPipeline(output_dir=str(output_dir))
    pipeline.load_data(
        phenotype_file=str(phe_file),
        genotype_file=str(geno_file),
        map_file=str(map_file),
        trait_columns=["Trait"],
        genotype_format="csv",
    )
    pipeline.align_samples()
    pipeline.run_analysis(
        traits=["Trait"],
        methods=["BAYESLOCO"],
        bayesloco_params={**_base_cfg(), "max_iter": 8, "patience": 3, "prior_tune_stage1_max_iter": 3, "prior_tune_stage2_max_iter": 4},
        outputs=["all_marker_pvalues"],
    )

    out_file = output_dir / "GWAS_Trait_all_results.csv"
    assert out_file.exists()
    out_df = pd.read_csv(out_file)
    assert "BAYESLOCO_P" in out_df.columns
    assert "BAYESLOCO_Effect" in out_df.columns
    meta_file = output_dir / "GWAS_Trait_BAYESLOCO_metadata.json"
    assert meta_file.exists()


def test_pipeline_bayesloco_preflight_rejects_invalid_config(tmp_path):
    phe, geno, map_df, _ = _tiny_inputs(n=20, m=40, seed=1001)
    pheno_df = pd.DataFrame({"ID": phe[:, 0], "Trait": phe[:, 1].astype(float)})
    geno_df = pd.DataFrame(geno, columns=map_df["SNP"])
    geno_df.insert(0, "ID", phe[:, 0])

    phe_file = tmp_path / "phe.csv"
    geno_file = tmp_path / "geno.csv"
    map_file = tmp_path / "map.csv"
    pheno_df.to_csv(phe_file, index=False)
    geno_df.to_csv(geno_file, index=False)
    map_df.to_csv(map_file, index=False)

    pipeline = GWASPipeline(output_dir=str(tmp_path / "out"))
    pipeline.load_data(
        phenotype_file=str(phe_file),
        genotype_file=str(geno_file),
        map_file=str(map_file),
        trait_columns=["Trait"],
        genotype_format="csv",
    )
    pipeline.align_samples()
    with pytest.raises(ValueError, match="unrelated_subset_indices"):
        pipeline.run_analysis(
            traits=["Trait"],
            methods=["BAYESLOCO"],
            bayesloco_params={"calibrate_stat_scale": "unrelated_subset"},
            outputs=["all_marker_pvalues"],
        )


def test_bayes_loco_deterministic_reproducible():
    phe, geno, map_df, cv = _tiny_inputs(n=28, m=72, seed=321)
    cfg = _base_cfg()
    res1 = PANICLE_BayesLOCO(phe=phe, geno=geno, map_data=map_df, CV=cv, verbose=False, bl_config=cfg)
    res2 = PANICLE_BayesLOCO(phe=phe, geno=geno, map_data=map_df, CV=cv, verbose=False, bl_config=cfg)
    np.testing.assert_allclose(res1.effects, res2.effects, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(res1.pvalues, res2.pvalues, atol=0.0, rtol=0.0)
    assert res1.metadata["prior_pi_selected"] == res2.metadata["prior_pi_selected"]
    assert res1.metadata["prior_slab_scale_selected"] == res2.metadata["prior_slab_scale_selected"]


def test_bayes_loco_prior_tuning_top_k_deterministic():
    phe, geno, map_df, cv = _tiny_inputs(n=40, m=96, seed=222)
    cfg = {
        **_base_cfg(),
        "prior_tune_pi_grid": (0.005, 0.02, 0.08),
        "prior_tune_slab_scale_grid": (0.75, 1.0),
        "prior_tune_two_stage": True,
        "prior_tune_top_k": 2,
        "prior_tune_stage1_max_iter": 4,
        "prior_tune_stage2_max_iter": 4,
        "prior_tune_prune_after_epochs": 2,
        "prior_tune_prune_rel_gap": 0.03,
        "max_iter": 8,
        "patience": 3,
    }
    res1 = PANICLE_BayesLOCO(phe=phe, geno=geno, map_data=map_df, CV=cv, verbose=False, bl_config=cfg)
    res2 = PANICLE_BayesLOCO(phe=phe, geno=geno, map_data=map_df, CV=cv, verbose=False, bl_config=cfg)
    assert res1.metadata["prior_pi_selected"] == res2.metadata["prior_pi_selected"]
    assert res1.metadata["prior_slab_scale_selected"] == res2.metadata["prior_slab_scale_selected"]
    assert int(res1.metadata["prior_tune_candidates_stage2"]) == 2


def test_bayes_loco_refine_cost_identity_and_unrelated_calibration():
    phe, geno, map_df, cv = _tiny_inputs(n=36, m=80, seed=777)
    unrelated_idx = list(range(18))
    cfg = {
        **_base_cfg(),
        "loco_mode": "refine",
        "loco_refine_iter": 3,
        "refine_patience": 2,
        "test_method": "wald",
        "robust_se": True,
        "calibrate_stat_scale": "unrelated_subset",
        "unrelated_subset_indices": unrelated_idx,
        "unrelated_subset_min_n": 10,
    }
    res = PANICLE_BayesLOCO(phe=phe, geno=geno, map_data=map_df, CV=cv, verbose=False, bl_config=cfg)
    md = res.metadata
    assert md["loco_mode"] == "refine"
    assert md["calibration_mode"] in {"unrelated_subset", "gc"}
    assert md["robust_se_applied"] is True
    assert md["pass_equiv_loco_refine"] > 0.0
    total = md["pass_equiv_prior_tune"] + md["pass_equiv_main_fit"] + md["pass_equiv_loco_refine"]
    assert md["pass_equiv_total"] == pytest.approx(total, rel=1e-8, abs=1e-8)


def test_bayes_loco_null_pvalues_are_not_inflated():
    n, m = 80, 160
    rng = np.random.default_rng(2026)
    ids = np.array([f"id{i:03d}" for i in range(n)])
    geno = rng.integers(0, 3, size=(n, m)).astype(np.int8)
    y = rng.normal(size=n)
    phe = np.column_stack([ids, y])
    cv = np.column_stack([rng.normal(size=n)]).astype(float)
    map_df = pd.DataFrame(
        {"SNP": [f"s{i}" for i in range(m)], "CHROM": [f"Chr{(i % 4) + 1:02d}" for i in range(m)], "POS": np.arange(1, m + 1)}
    )
    cfg = {**_base_cfg(), "max_iter": 10, "prior_tune_stage1_max_iter": 3, "prior_tune_stage2_max_iter": 4}
    res = PANICLE_BayesLOCO(phe=phe, geno=geno, map_data=map_df, CV=cv, verbose=False, bl_config=cfg)
    frac = float(np.mean(res.pvalues < 0.05))
    assert 0.005 <= frac <= 0.15
    assert 0.25 <= float(np.median(res.pvalues)) <= 0.75


def test_bayes_loco_detects_strong_signal():
    n, m = 90, 180
    rng = np.random.default_rng(2027)
    ids = np.array([f"id{i:03d}" for i in range(n)])
    geno = rng.integers(0, 3, size=(n, m)).astype(np.int8)
    causal = 25
    y = 1.4 * geno[:, causal].astype(float) + rng.normal(scale=1.0, size=n)
    phe = np.column_stack([ids, y])
    cv = np.column_stack([rng.normal(size=n)]).astype(float)
    map_df = pd.DataFrame(
        {"SNP": [f"s{i}" for i in range(m)], "CHROM": [f"Chr{(i % 5) + 1:02d}" for i in range(m)], "POS": np.arange(1, m + 1)}
    )
    cfg = {**_base_cfg(), "max_iter": 12, "prior_tune_stage1_max_iter": 4, "prior_tune_stage2_max_iter": 5}
    res = PANICLE_BayesLOCO(phe=phe, geno=geno, map_data=map_df, CV=cv, verbose=False, bl_config=cfg)
    rank = int(np.where(np.argsort(res.pvalues) == causal)[0][0]) + 1
    assert rank <= 10
