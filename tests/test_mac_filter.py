"""Tests for the per-trait minor allele count (MAC) filter."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panicle.pipelines.gwas import GWASPipeline
from panicle.utils.data_types import (
    AssociationResults,
    GenotypeMap,
    GenotypeMatrix,
)
from panicle.utils.stats import (
    compute_mac_keep_indices,
    pad_association_results,
)


def test_compute_mac_keep_indices_disabled_returns_none() -> None:
    g = np.zeros((10, 5), dtype=np.int8)
    assert compute_mac_keep_indices(g, 0) is None
    assert compute_mac_keep_indices(g, None) is None


def test_compute_mac_keep_indices_drops_rare_markers() -> None:
    n_ind = 50
    n_mrk = 6
    g = np.zeros((n_ind, n_mrk), dtype=np.int8)
    # Marker 0: singleton alt (MAC=1)
    g[0, 0] = 2
    # Marker 1: doubleton via two hets (MAC=2)
    g[0, 1] = 1
    g[1, 1] = 1
    # Marker 2: MAC=5 (five hets)
    for i in range(5):
        g[i, 2] = 1
    # Marker 3: MAC=10 (five homozygous alt)
    for i in range(5):
        g[i, 3] = 2
    # Marker 4: monomorphic reference (MAC=0)
    pass
    # Marker 5: common variant (MAC=50)
    g[:25, 5] = 2

    # With min_mac=5, keep markers 2, 3, 5
    keep = compute_mac_keep_indices(g, 5)
    assert keep.tolist() == [2, 3, 5]

    # With min_mac=10, only markers 3 and 5 survive (marker 2 has MAC=5)
    keep10 = compute_mac_keep_indices(g, 10)
    assert keep10.tolist() == [3, 5]


def test_compute_mac_keep_indices_works_on_genotype_matrix() -> None:
    n_ind, n_mrk = 30, 4
    g = np.zeros((n_ind, n_mrk), dtype=np.int8)
    g[0, 0] = 2  # singleton
    g[:15, 1] = 2  # MAC = 30
    gm = GenotypeMatrix(g, is_imputed=True, precompute_alleles=False)
    keep = compute_mac_keep_indices(gm, 5)
    # Marker 0 dropped, 1 kept, 2 & 3 monomorphic (MAC=0) dropped
    assert keep.tolist() == [1]


def test_pad_association_results_noop_when_no_filter() -> None:
    res = AssociationResults(
        effects=np.array([0.1, 0.2, 0.3]),
        se=np.array([0.01, 0.02, 0.03]),
        pvalues=np.array([0.5, 0.01, 0.9]),
    )
    out = pad_association_results(res, None, 3)
    assert out is res  # no-op returns original


def test_pad_association_results_expands_with_nan() -> None:
    # 5-marker map, only indices [1, 3] scanned
    res = AssociationResults(
        effects=np.array([0.2, 0.4]),
        se=np.array([0.02, 0.04]),
        pvalues=np.array([0.01, 0.001]),
    )
    keep = np.array([1, 3], dtype=np.int64)
    out = pad_association_results(res, keep, 5)
    assert out is not res
    assert len(out.pvalues) == 5
    assert np.isnan(out.pvalues[0])
    assert out.pvalues[1] == pytest.approx(0.01)
    assert np.isnan(out.pvalues[2])
    assert out.pvalues[3] == pytest.approx(0.001)
    assert np.isnan(out.pvalues[4])
    assert out.effects[1] == pytest.approx(0.2)
    assert out.se[3] == pytest.approx(0.04)


def test_genotype_matrix_subset_markers_boolean_and_int() -> None:
    rng = np.random.default_rng(0)
    g = rng.integers(0, 3, size=(20, 10)).astype(np.int8)
    gm = GenotypeMatrix(g, is_imputed=True, precompute_alleles=False)

    mask = np.zeros(10, dtype=bool)
    mask[[2, 5, 8]] = True
    sub = gm.subset_markers(mask)
    assert sub.shape == (20, 3)
    assert np.array_equal(sub.to_numpy(), g[:, mask])

    sub_int = gm.subset_markers(np.array([1, 4, 9]))
    assert sub_int.shape == (20, 3)
    assert np.array_equal(sub_int.to_numpy(), g[:, [1, 4, 9]])


def test_genotype_map_subset_markers_preserves_columns() -> None:
    df = pd.DataFrame({
        'SNP': [f'm{i}' for i in range(8)],
        'Marker_ID': [f'm{i}' for i in range(8)],
        'CHROM': ['1'] * 4 + ['2'] * 4,
        'POS': np.arange(8) * 1000,
    })
    gmap = GenotypeMap(df)
    keep = np.array([0, 2, 5, 7], dtype=np.int64)
    sub = gmap.subset_markers(keep)
    assert sub.n_markers == 4
    assert list(sub.marker_ids.values) == ['m0', 'm2', 'm5', 'm7']
    assert list(sub.chromosomes.values) == ['1', '1', '2', '2']


@pytest.fixture
def singleton_dataset(tmp_path: Path):
    """Synthetic dataset with a deliberate singleton marker that would drive a
    spurious significant p-value without the MAC filter."""
    rng = np.random.default_rng(123)
    n_samples = 60
    n_markers = 30

    sample_ids = [f"S{i:03d}" for i in range(n_samples)]

    # Make trait strongly correlated with sample index (so the single individual
    # carrying the singleton happens to be an extreme one).
    trait = np.linspace(-3.0, 3.0, n_samples) + rng.standard_normal(n_samples) * 0.1

    pheno = pd.DataFrame({'ID': sample_ids, 'trait': trait})
    pheno_file = tmp_path / "phenotypes.csv"
    pheno.to_csv(pheno_file, index=False)

    # Random common markers + one singleton at the extreme sample
    g = rng.integers(0, 3, size=(n_samples, n_markers)).astype(np.int8)
    # Marker 0: singleton in the extreme sample (last one, trait = +3)
    g[:, 0] = 0
    g[-1, 0] = 2

    # Marker 1: doubleton in the two most extreme samples
    g[:, 1] = 0
    g[-1, 1] = 2
    g[-2, 1] = 2

    marker_ids = [f"SNP{i:04d}" for i in range(n_markers)]
    geno_df = pd.DataFrame(g, columns=marker_ids)
    geno_df.insert(0, 'ID', sample_ids)
    geno_file = tmp_path / "genotypes.csv"
    geno_df.to_csv(geno_file, index=False)

    map_df = pd.DataFrame({
        'SNP': marker_ids,
        'CHROM': ['1'] * n_markers,
        'POS': [i * 1000 for i in range(n_markers)],
    })
    map_file = tmp_path / "map.csv"
    map_df.to_csv(map_file, index=False)

    return {
        'phenotype_file': pheno_file,
        'genotype_file': geno_file,
        'map_file': map_file,
        'n_samples': n_samples,
        'n_markers': n_markers,
    }


def test_pipeline_mac_filter_drops_singleton_in_output(singleton_dataset, tmp_path):
    """With min_mac=5, the singleton/doubleton markers get NaN p-values in
    the per-trait results table (padded back to full-map length)."""

    pipeline = GWASPipeline(output_dir=str(tmp_path / "out_filtered"))
    pipeline.load_data(
        phenotype_file=str(singleton_dataset['phenotype_file']),
        genotype_file=str(singleton_dataset['genotype_file']),
        map_file=str(singleton_dataset['map_file']),
        trait_columns=['trait'],
        genotype_format='csv',
    )
    pipeline.align_samples()
    pipeline.run_analysis(
        traits=['trait'],
        methods=['GLM'],
        min_mac=5,
        outputs=['all_marker_pvalues'],
    )

    out = pd.read_csv(tmp_path / "out_filtered" / "GWAS_trait_all_results.csv")
    # Full map preserved in output
    assert len(out) == singleton_dataset['n_markers']
    # Singleton marker should have NaN p-value (padded after the filter drop)
    assert pd.isna(out.loc[out['SNP'] == 'SNP0000', 'GLM_P'].iloc[0])
    # Doubleton marker also dropped (MAC=4 < 5)
    assert pd.isna(out.loc[out['SNP'] == 'SNP0001', 'GLM_P'].iloc[0])


def test_pipeline_mac_filter_disabled_keeps_all_markers(singleton_dataset, tmp_path):
    """With min_mac=0, p-values are produced for every marker (baseline)."""

    pipeline = GWASPipeline(output_dir=str(tmp_path / "out_unfiltered"))
    pipeline.load_data(
        phenotype_file=str(singleton_dataset['phenotype_file']),
        genotype_file=str(singleton_dataset['genotype_file']),
        map_file=str(singleton_dataset['map_file']),
        trait_columns=['trait'],
        genotype_format='csv',
    )
    pipeline.align_samples()
    pipeline.run_analysis(
        traits=['trait'],
        methods=['GLM'],
        min_mac=0,
        outputs=['all_marker_pvalues'],
    )

    out = pd.read_csv(tmp_path / "out_unfiltered" / "GWAS_trait_all_results.csv")
    assert len(out) == singleton_dataset['n_markers']
    assert not out['GLM_P'].isna().any()
