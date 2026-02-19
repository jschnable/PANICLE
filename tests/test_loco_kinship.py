import numpy as np
import pandas as pd

from panicle.utils.data_types import GenotypeMatrix
from panicle.matrix.kinship import PANICLE_K_VanRaden
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.association.mlm import PANICLE_MLM
from panicle.association.mlm_loco import PANICLE_MLM_LOCO


def _make_test_data(seed: int = 123):
    rng = np.random.default_rng(seed)
    n_individuals = 12
    n_markers = 30

    genotypes = rng.integers(0, 3, size=(n_individuals, n_markers), dtype=np.int8)
    chroms = np.array(["Chr1"] * 10 + ["Chr2"] * 10 + ["Chr3"] * 10)

    map_df = pd.DataFrame({
        "SNP": [f"SNP{i:04d}" for i in range(n_markers)],
        "CHROM": chroms,
        "POS": np.arange(n_markers) * 100 + 1,
    })

    return genotypes, map_df


def test_loco_kinship_matches_naive():
    genotypes, map_df = _make_test_data()
    geno = GenotypeMatrix(genotypes)

    loco = PANICLE_K_VanRaden_LOCO(geno, map_df, maxLine=7, verbose=False)

    full_ref = PANICLE_K_VanRaden(geno, maxLine=7, verbose=False).to_numpy()
    full_loco = loco.get_full().to_numpy()
    # Tolerance relaxed slightly for float32 kinship computation
    np.testing.assert_allclose(full_loco, full_ref, rtol=1e-6, atol=1e-6)

    chroms = map_df["CHROM"].to_numpy()
    for chrom in np.unique(chroms):
        keep_mask = chroms != chrom
        geno_subset = genotypes[:, keep_mask]
        ref = PANICLE_K_VanRaden(geno_subset, maxLine=7, verbose=False).to_numpy()
        loco_k = loco.get_loco(chrom).to_numpy()
        np.testing.assert_allclose(loco_k, ref, rtol=1e-8, atol=5e-7)


def test_vanraden_numpy_missing_matches_genotype_matrix_missing():
    rng = np.random.default_rng(222)
    geno = rng.integers(0, 3, size=(18, 40), dtype=np.int8)
    geno[rng.random(geno.shape) < 0.12] = -9

    k_np = PANICLE_K_VanRaden(geno, maxLine=9, verbose=False).to_numpy()
    k_gm = PANICLE_K_VanRaden(GenotypeMatrix(geno), maxLine=9, verbose=False).to_numpy()

    np.testing.assert_allclose(k_np, k_gm, rtol=1e-8, atol=1e-8)


def test_mlm_loco_matches_per_chrom_mlm():
    rng = np.random.default_rng(321)
    n_individuals = 20
    n_markers = 24

    genotypes = rng.integers(0, 3, size=(n_individuals, n_markers), dtype=np.int8)
    chroms = np.array(["Chr1"] * 8 + ["Chr2"] * 8 + ["Chr3"] * 8)
    map_df = pd.DataFrame({
        "SNP": [f"SNP{i:04d}" for i in range(n_markers)],
        "CHROM": chroms,
        "POS": np.arange(n_markers) * 50 + 1,
    })

    phe = np.column_stack([np.arange(n_individuals), rng.normal(size=n_individuals)])

    geno = GenotypeMatrix(genotypes)
    loco = PANICLE_K_VanRaden_LOCO(geno, map_df, maxLine=6, verbose=False)

    loco_results = PANICLE_MLM_LOCO(
        phe=phe,
        geno=geno,
        map_data=map_df,
        loco_kinship=loco,
        maxLine=6,
        cpu=1,
        verbose=False,
    )

    expected_effects = np.zeros(n_markers, dtype=np.float64)
    expected_se = np.zeros(n_markers, dtype=np.float64)
    expected_pvals = np.ones(n_markers, dtype=np.float64)

    for chrom in loco.chromosomes:
        indices = np.where(chroms == chrom)[0]
        geno_subset = genotypes[:, indices]
        res = PANICLE_MLM(
            phe=phe,
            geno=geno_subset,
            K=loco.get_loco(chrom),
            eigenK=loco.get_eigen(chrom),
            maxLine=6,
            cpu=1,
            verbose=False,
        )
        expected_effects[indices] = res.effects
        expected_se[indices] = res.se
        expected_pvals[indices] = res.pvalues

    np.testing.assert_allclose(loco_results.effects, expected_effects, rtol=1e-8, atol=1e-8, equal_nan=True)
    np.testing.assert_allclose(loco_results.se, expected_se, rtol=1e-8, atol=1e-8, equal_nan=True)
    np.testing.assert_allclose(loco_results.pvalues, expected_pvals, rtol=1e-8, atol=1e-8, equal_nan=True)
