"""Correctness tests for LD-aware structure, locus identification, lambda_1000
and REML heritability, on simulations with known truth."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panicle.matrix.admixture import fit_admixture
from panicle.matrix.kinship import PANICLE_K_VanRaden
from panicle.matrix.ld import ld_prune
from panicle.matrix.pca import PANICLE_PCA
from panicle.postgwas import (
    case_control_counts,
    gower_scale,
    identify_loci,
    lambda_1000,
    reml_heritability,
)

from sim_structured import (
    marginal_pvalues,
    polygenic_phenotype,
    simulate_structured,
    simulate_unstructured,
    standardize,
)


def _r2_explained(Y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    resid = Y - X1 @ beta
    return 1 - resid.var(axis=0) / Y.var(axis=0)


# ---------------------------------------------------------------------------
# LD pruning and structure
# ---------------------------------------------------------------------------

def test_ld_prune_leaves_no_pair_above_threshold():
    d = simulate_unstructured(n=200, n_chrom=2, snps_per_chrom=400, rho=0.95, seed=3)
    kept = ld_prune(d.genotypes, d.chrom, d.pos, r2_threshold=0.2, window_kb=50,
                    max_window_snps=1000, chunk=64)
    assert 0 < kept.size < d.genotypes.shape[1]
    X = standardize(d.genotypes[:, kept])
    R2 = (X.T @ X / X.shape[0]) ** 2
    ch, ps = d.chrom[kept], d.pos[kept]
    for i, j in itertools.combinations(range(kept.size), 2):
        if ch[i] == ch[j] and abs(ps[i] - ps[j]) <= 50_000:
            assert R2[i, j] <= 0.2 + 1e-4


def test_ld_prune_drops_rare_markers():
    d = simulate_unstructured(n=200, n_chrom=1, snps_per_chrom=200, seed=4)
    G = d.genotypes.copy()
    G[:, 10] = 0
    G[0, 10] = 1  # MAF 0.0025
    kept = ld_prune(G, d.chrom, d.pos, min_maf=0.01)
    assert 10 not in kept


@pytest.fixture(scope="module")
def inversion_panel():
    return simulate_structured(n=400, n_chrom=4, snps_per_chrom=2000,
                               inversion=(0, 400, 1200), seed=1)


def test_pruned_pca_tracks_ancestry_not_inversion(inversion_panel):
    d = inversion_panel
    pcs_all = PANICLE_PCA(M=d.genotypes.astype(float), pcs_keep=3, verbose=False)
    kept = ld_prune(d.genotypes, d.chrom, d.pos, r2_threshold=0.2, window_kb=500)
    pcs_pruned = PANICLE_PCA(M=d.genotypes[:, kept].astype(float), pcs_keep=3, verbose=False)

    inv_corr = lambda pcs: [abs(np.corrcoef(pcs[:, k], d.inversion_dosage)[0, 1]) for k in range(3)]
    # Without pruning the inversion's long LD block becomes PC1.
    assert inv_corr(pcs_all)[0] > 0.9
    assert _r2_explained(d.Q[:, :2], pcs_all[:, :2]).min() < 0.6
    # With in-sample pruning the inversion collapses to a few markers.
    assert np.isin(d.inversion_markers, kept).sum() < 20
    assert max(inv_corr(pcs_pruned)) < 0.3
    assert _r2_explained(d.Q[:, :2], pcs_pruned[:, :2]).min() > 0.85


def test_admixture_recovers_true_ancestry(inversion_panel):
    d = inversion_panel
    kept = ld_prune(d.genotypes, d.chrom, d.pos, r2_threshold=0.2, window_kb=500)
    fit = fit_admixture(d.genotypes, 3, marker_indices=kept, n_restarts=1, seed=0)
    np.testing.assert_allclose(fit.Q.sum(axis=1), 1.0, atol=1e-8)
    assert fit.covariates().shape == (d.genotypes.shape[0], 2)
    C = np.corrcoef(fit.Q.T, d.Q.T)[:3, 3:]
    best = max(sum(C[i, perm[i]] for i in range(3)) / 3 for perm in itertools.permutations(range(3)))
    assert best > 0.9


def test_admixture_tolerates_missing_calls():
    d = simulate_structured(n=200, n_chrom=2, snps_per_chrom=600, rho=0.0, fst=0.15, seed=9)
    G = d.genotypes.astype(float)
    rng = np.random.default_rng(0)
    G[rng.random(G.shape) < 0.05] = -9
    fit = fit_admixture(G, 3, n_restarts=1, seed=1)
    C = np.corrcoef(fit.Q.T, d.Q.T)[:3, 3:]
    best = max(sum(C[i, perm[i]] for i in range(3)) / 3 for perm in itertools.permutations(range(3)))
    assert best > 0.85
    assert np.isfinite(fit.log_likelihood)


def test_admixture_rejects_k_below_two():
    with pytest.raises(ValueError):
        fit_admixture(np.zeros((5, 5)), 1)


# ---------------------------------------------------------------------------
# lambda_1000
# ---------------------------------------------------------------------------

def test_lambda_1000_quantitative_and_case_control():
    assert lambda_1000(1.10, 500) == pytest.approx(1.20)
    assert lambda_1000(1.10, 4000) == pytest.approx(1.025)
    assert lambda_1000(1.0, 123) == pytest.approx(1.0)
    # 1000 cases + 1000 controls is the reference: unchanged.
    assert lambda_1000(1.2, n_cases=1000, n_controls=1000) == pytest.approx(1.2)
    assert lambda_1000(1.2, n_cases=250, n_controls=750) == pytest.approx(1 + 0.2 * (1 / 250 + 1 / 750) * 500)
    assert np.isnan(lambda_1000(float("nan"), 100))
    assert np.isnan(lambda_1000(1.1, 0))


def test_case_control_detection():
    assert case_control_counts(np.array([0, 1, 1, 0, 1, np.nan])) == (3, 2)
    assert case_control_counts(np.array([1, 2, 2])) == (2, 1)
    assert case_control_counts(np.array([0.1, 1.5, 2.0])) == (None, None)


def test_lambda_1000_matches_simulated_polygenic_scaling():
    """lambda-1 grows ~linearly in N under polygenicity; lambda_1000 should not."""
    from panicle.utils.stats import genomic_inflation_factor
    d = simulate_unstructured(n=3000, n_chrom=5, snps_per_chrom=800, seed=21)
    y = polygenic_phenotype(d.genotypes, 0.6, np.random.default_rng(21))
    lam = {}
    for n in (1000, 3000):
        p = marginal_pvalues(d.genotypes[:n], y[:n])
        lam[n] = genomic_inflation_factor(p)
    assert lam[3000] - 1 > 1.8 * (lam[1000] - 1)
    l1000 = [lambda_1000(lam[n], n) for n in (1000, 3000)]
    assert abs(l1000[0] - l1000[1]) < 0.5 * (lam[3000] - lam[1000])


# ---------------------------------------------------------------------------
# Locus identification
# ---------------------------------------------------------------------------

def test_identify_loci_finds_each_qtl_once():
    d = simulate_unstructured(n=800, n_chrom=4, snps_per_chrom=600, rho=0.95, seed=13)
    rng = np.random.default_rng(13)
    qtl = [150, 600 + 300, 3 * 600 + 450]
    X = standardize(d.genotypes)
    y = X[:, qtl] @ np.array([0.5, -0.45, 0.4]) + rng.standard_normal(800)
    p = marginal_pvalues(d.genotypes, y)
    threshold = 0.05 / p.size
    loci = identify_loci(p, d.chrom, d.pos, d.genotypes, p_threshold=threshold,
                         clump_kb=250, clump_r2=0.1, effects=np.zeros(p.size))
    assert len(loci) >= 3
    leads = loci["Lead_POS"].to_numpy()
    for q in qtl:
        same_chr = loci["CHROM"].to_numpy() == d.chrom[q]
        dist = np.abs(leads[same_chr] - d.pos[q])
        assert dist.min() <= 20_000
        # every QTL lies inside a locus span
        inside = (loci["Start"] <= d.pos[q]) & (loci["End"] >= d.pos[q]) & (loci["CHROM"] == d.chrom[q])
        assert inside.sum() == 1
    # Strong loci clump many linked significant markers, not one locus per SNP.
    n_sig = int((p <= threshold).sum())
    assert len(loci) < n_sig
    assert loci["Lead_P"].is_monotonic_increasing


def test_identify_loci_merge_gap_and_empty():
    d = simulate_unstructured(n=300, n_chrom=1, snps_per_chrom=300, rho=0.0, seed=2)
    p = np.ones(300)
    p[[50, 60]] = 1e-12  # independent markers 10 kb apart
    loci = identify_loci(p, d.chrom, d.pos, d.genotypes, p_threshold=1e-8, clump_r2=0.5)
    assert len(loci) == 2
    merged = identify_loci(p, d.chrom, d.pos, d.genotypes, p_threshold=1e-8, clump_r2=0.5,
                           merge_gap_kb=20)
    assert len(merged) == 1 and merged.loc[0, "N_Significant"] == 2
    none = identify_loci(np.ones(300), d.chrom, d.pos, d.genotypes, p_threshold=1e-8)
    assert none.empty


# ---------------------------------------------------------------------------
# REML heritability (null model, global kinship)
# ---------------------------------------------------------------------------

def _kinship(G):
    K = PANICLE_K_VanRaden(G.astype(float), verbose=False)
    return np.asarray(K.to_numpy() if hasattr(K, "to_numpy") else K)


def _reml_loglik(vg, ve, y, X, K):
    V = vg * K + ve * np.eye(len(y))
    Vi = np.linalg.inv(V)
    XViX = X.T @ Vi @ X
    P = Vi - Vi @ X @ np.linalg.solve(XViX, X.T @ Vi)
    return -0.5 * (np.linalg.slogdet(V)[1] + np.linalg.slogdet(XViX)[1] + y @ P @ y)


def _grm_consistent_phenotype(G, h2, rng):
    """g = Z u on centred (unscaled) genotypes: the model VanRaden K assumes."""
    Z = G.astype(float)
    Z -= Z.mean(axis=0)
    g = Z @ rng.standard_normal(Z.shape[1])
    g = (g - g.mean()) / g.std() * np.sqrt(h2)
    return g + rng.standard_normal(len(g)) * np.sqrt(1 - h2)


def test_reml_h2_matches_brute_force_reml_and_numerical_se():
    from scipy import optimize
    d = simulate_unstructured(n=300, n_chrom=3, snps_per_chrom=400, seed=3)
    rng = np.random.default_rng(3)
    y = _grm_consistent_phenotype(d.genotypes, 0.5, rng)
    C = rng.standard_normal((300, 2))
    K = _kinship(d.genotypes)
    X = np.column_stack([np.ones(300), C])
    res = reml_heritability(y, K, C)

    opt = optimize.minimize(
        lambda t: -_reml_loglik(np.exp(t[0]), np.exp(t[1]), y, X, K),
        [np.log(0.5), np.log(0.5)], method="Nelder-Mead",
        options=dict(xatol=1e-9, fatol=1e-12, maxiter=4000),
    )
    vg, ve = np.exp(opt.x)
    assert res.h2 == pytest.approx(vg / (vg + ve), abs=1e-4)
    assert res.vg == pytest.approx(vg, rel=1e-3) and res.ve == pytest.approx(ve, rel=1e-3)

    # SE: information-matrix SE agrees with the observed (numerical Hessian) SE.
    f = lambda a, b: _reml_loglik(a, b, y, X, K)
    th = np.array([res.vg, res.ve])
    e = 1e-3
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * e, np.eye(2)[j] * e
            H[i, j] = (f(*(th + ei + ej)) - f(*(th + ei - ej))
                       - f(*(th - ei + ej)) + f(*(th - ei - ej))) / (4 * e * e)
    grad = np.array([res.ve, -res.vg]) / (res.vg + res.ve) ** 2
    se_obs = np.sqrt(grad @ np.linalg.inv(-H) @ grad)
    assert res.h2_se == pytest.approx(se_obs, rel=0.1)
    assert res.lrt_p < 1e-3


@pytest.mark.parametrize("inbred", [False, True])
def test_reml_h2_recovers_simulated_heritability(inbred):
    est, ses = [], []
    for seed in range(40, 52):
        G = simulate_unstructured(n=400, n_chrom=4, snps_per_chrom=400, seed=seed).genotypes
        if inbred:
            G = (G >= 1).astype(np.int8) * 2  # homozygous lines
        y = _grm_consistent_phenotype(G, 0.4, np.random.default_rng(seed))
        r = reml_heritability(y, _kinship(G))
        est.append(r.h2)
        ses.append(r.h2_se)
    est = np.asarray(est)
    # unbiased within Monte-Carlo error, and the SE is not anti-conservative
    assert abs(est.mean() - 0.4) < 3 * est.std() / np.sqrt(est.size) + 0.02
    assert np.mean(ses) > 0.8 * est.std()


def test_reml_h2_null_trait_and_collinear_covariates():
    d = simulate_unstructured(n=300, n_chrom=3, snps_per_chrom=300, seed=8)
    rng = np.random.default_rng(8)
    y = rng.standard_normal(300)
    c = rng.standard_normal(300)
    covs = np.column_stack([c, 2 * c, np.ones(300)])  # duplicates + intercept
    r = reml_heritability(y, _kinship(d.genotypes), covs)
    assert r.n_fixed == 2
    assert r.h2 < 0.25
    assert r.lrt_p > 0.01


def test_gower_scale():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 20))
    K = gower_scale(A @ A.T * 3.7)
    assert np.trace(K) / 50 - K.mean() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End to end through the CLI
# ---------------------------------------------------------------------------

def test_cli_end_to_end_structure_loci_lambda1000_h2(tmp_path: Path):
    from panicle.cli.gwas import main

    d = simulate_structured(n=300, n_chrom=3, snps_per_chrom=1200,
                            inversion=(0, 300, 600), seed=5)
    rng = np.random.default_rng(5)
    m = d.genotypes.shape[1]
    qtl = [1200 + 600, 2 * 1200 + 900]
    X = standardize(d.genotypes)
    y = X[:, qtl] @ np.array([0.7, -0.7]) + 1.0 * d.Q[:, 0] + rng.standard_normal(300) * 0.8
    ids = [f"L{i:03d}" for i in range(300)]
    snps = [f"S{j}" for j in range(m)]
    pd.DataFrame({"ID": ids, "Yield": y}).to_csv(tmp_path / "ph.csv", index=False)
    g = pd.DataFrame(d.genotypes, columns=snps)
    g.insert(0, "ID", ids)
    g.to_csv(tmp_path / "g.csv", index=False)
    pd.DataFrame({"SNP": snps, "CHROM": d.chrom, "POS": d.pos}).to_csv(tmp_path / "map.csv", index=False)

    out = tmp_path / "out"
    rc = main([
        "--phenotype", str(tmp_path / "ph.csv"), "--genotype", str(tmp_path / "g.csv"),
        "--map", str(tmp_path / "map.csv"), "--format", "csv", "--outputdir", str(out),
        "--methods", "GLM,MLM", "--n-pcs", "2", "--pca-ld-prune", "--admixture-k", "3",
        "--outputs", "all_marker_pvalues",
        "significant_marker_pvalues",
    ])
    assert rc == 0

    cov = pd.read_csv(out / "population_structure_covariates.csv")
    assert {"ID", "PC1", "PC2", "Q1", "Q2", "Q1_full", "Q2_full", "Q3_full"} <= set(cov.columns)
    assert (out / "structure_pruned_markers.csv").exists()

    summary = pd.read_csv(out / "GWAS_summary_by_traits_methods.csv")
    for col in ("Lambda_1000", "N_Loci", "H2_REML", "H2_REML_SE"):
        assert col in summary.columns
    glm = summary.set_index("Method").loc["GLM"]
    assert glm["Lambda_1000"] == pytest.approx(lambda_1000(glm["Lambda_GC"], 300), abs=2e-3)
    assert 0 <= glm["H2_REML"] <= 1 and glm["H2_REML_SE"] > 0
    # one value per trait, repeated on every method row
    assert summary["H2_REML"].nunique() == 1

    loci = pd.read_csv(out / "GWAS_Yield_GLM_loci.csv")
    assert len(loci) == glm["N_Loci"]
    assert (out / "LD_decay.csv").exists()
    for q in qtl:
        hit = (loci["CHROM"] == d.chrom[q]) & (loci["Start"] <= d.pos[q] + 20_000) & (loci["End"] >= d.pos[q] - 20_000)
        assert hit.any(), f"QTL {snps[q]} not recovered"
    h2_table = pd.read_csv(out / "GWAS_Yield_heritability.csv")
    assert {"h2", "h2_se", "vg", "ve", "lrt_p", "n", "n_fixed"} <= set(h2_table.columns)
    assert h2_table.loc[0, "h2"] == pytest.approx(glm["H2_REML"], abs=1e-4)
    assert h2_table.loc[0, "n_fixed"] == 1 + 2 + 2  # intercept + 2 PCs + 2 Q


def test_pipeline_requires_map_for_pruning(tmp_path: Path):
    from panicle.pipelines.gwas import GWASPipeline
    from panicle.utils.data_types import GenotypeMatrix

    p = GWASPipeline(str(tmp_path))
    p.genotype_matrix = GenotypeMatrix(np.random.default_rng(0).integers(0, 3, (20, 30)).astype(np.int8))
    p._matched_indices = np.arange(20)
    p.geno_map = None
    with pytest.raises(ValueError, match="genetic map"):
        p.compute_population_structure(n_pcs=2, calculate_kinship=False, ld_prune_pca=True)
    with pytest.raises(ValueError, match="admixture_k"):
        p.compute_population_structure(n_pcs=2, calculate_kinship=False, admixture_k=1)


def test_loco_full_kinship_equals_global_vanraden():
    """LOCO-MLM runs reuse LocoKinship.get_full() for REML h2: it must be the global K."""
    from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
    from panicle.utils.data_types import GenotypeMap

    d = simulate_unstructured(n=120, n_chrom=3, snps_per_chrom=200, seed=6)
    gmap = GenotypeMap(pd.DataFrame({
        "SNP": [f"S{j}" for j in range(d.genotypes.shape[1])], "CHROM": d.chrom, "POS": d.pos,
    }))
    loco = PANICLE_K_VanRaden_LOCO(d.genotypes.astype(float), gmap, verbose=False)
    np.testing.assert_allclose(loco.get_full().to_numpy(), _kinship(d.genotypes), atol=1e-5)


# ---------------------------------------------------------------------------
# LD decay and automatic clumping window
# ---------------------------------------------------------------------------

def test_ld_decay_matches_brute_force_pairs():
    from panicle.matrix.ld import ld_decay
    d = simulate_unstructured(n=200, n_chrom=2, snps_per_chrom=150, rho=0.95, seed=4)
    dec = ld_decay(d.genotypes, d.chrom, d.pos, max_kb=60, min_kb=1, n_bins=12,
                   max_window_snps=1000, min_maf=0.0, chunk=37)
    X = standardize(d.genotypes)
    R2 = (X.T @ X / X.shape[0]) ** 2
    edges = np.concatenate([[0.0], np.geomspace(1, 60, 12)]) * 1000
    sums = np.zeros(12)
    counts = np.zeros(12, dtype=int)
    for i, j in itertools.combinations(range(X.shape[1]), 2):
        dist = abs(d.pos[j] - d.pos[i])
        if d.chrom[i] != d.chrom[j] or dist > 60_000:
            continue
        b = min(np.searchsorted(edges, dist, side="right") - 1, 11)
        sums[b] += R2[i, j]
        counts[b] += 1
    np.testing.assert_array_equal(dec.n_pairs, counts)
    filled = counts > 0
    np.testing.assert_allclose(dec.mean_r2[filled], sums[filled] / counts[filled], atol=1e-4)


def test_ld_decay_distance_tracks_ld_extent():
    from panicle.matrix.ld import ld_decay
    short = simulate_unstructured(n=300, n_chrom=2, snps_per_chrom=600, rho=0.9, spacing_bp=1000, seed=1)
    long_ = simulate_unstructured(n=300, n_chrom=2, snps_per_chrom=600, rho=0.99, spacing_bp=10000, seed=1)
    d_short = ld_decay(short.genotypes, short.chrom, short.pos, max_kb=5000).decay_distance_kb(0.1)
    d_long = ld_decay(long_.genotypes, long_.chrom, long_.pos, max_kb=5000).decay_distance_kb(0.1)
    assert d_short < 30
    assert 300 < d_long < 3000


def test_auto_clump_window_does_not_split_long_ld_block():
    from panicle.matrix.ld import ld_decay
    d = simulate_unstructured(n=600, n_chrom=2, snps_per_chrom=600, rho=0.99, spacing_bp=10000, seed=3)
    rng = np.random.default_rng(3)
    X = standardize(d.genotypes)
    y = 0.6 * X[:, 300] + rng.standard_normal(600)   # one QTL inside a long-LD chromosome
    p = marginal_pvalues(d.genotypes, y)
    thr = 0.05 / p.size
    fixed = identify_loci(p, d.chrom, d.pos, d.genotypes, p_threshold=thr, clump_kb=250)
    window = ld_decay(d.genotypes, d.chrom, d.pos, max_kb=5000).decay_distance_kb(0.1)
    auto = identify_loci(p, d.chrom, d.pos, d.genotypes, p_threshold=thr, clump_kb=window)
    assert len(fixed) > 1          # 250 kb splits the single signal
    assert len(auto) == 1          # LD-decay window keeps it whole
    assert auto.loc[0, "Start"] <= d.pos[300] <= auto.loc[0, "End"]
