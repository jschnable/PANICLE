"""Integration tests for GWASPipeline end-to-end workflows."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panicle.pipelines.gwas import GWASPipeline


@pytest.fixture
def synthetic_data(tmp_path: Path):
    """Create small synthetic genotype and phenotype files for testing."""

    # Create phenotype file: 20 samples, 2 traits
    rng = np.random.default_rng(42)
    n_samples = 20

    sample_ids = [f"Sample{i:03d}" for i in range(n_samples)]
    trait1 = rng.standard_normal(n_samples) * 10 + 50  # Mean ~50
    trait2 = rng.standard_normal(n_samples) * 5 + 20   # Mean ~20

    pheno_df = pd.DataFrame({
        'ID': sample_ids,
        'Height': trait1,
        'Yield': trait2
    })

    pheno_file = tmp_path / "phenotypes.csv"
    pheno_df.to_csv(pheno_file, index=False)

    # Create genotype file: 20 samples, 50 markers
    n_markers = 50
    marker_names = [f"SNP{i:04d}" for i in range(n_markers)]

    # Generate random genotypes (0, 1, 2)
    genotypes = rng.integers(0, 3, size=(n_samples, n_markers))

    # Add a few markers with stronger effects for trait1 to ensure some signal
    # Markers 10-12 will have correlation with Height
    for i in range(10, 13):
        genotypes[:, i] = (trait1 > 50).astype(int) + rng.integers(0, 2, n_samples)
        genotypes[:, i] = np.clip(genotypes[:, i], 0, 2)

    geno_df = pd.DataFrame(genotypes, columns=marker_names)
    geno_df.insert(0, 'ID', sample_ids)

    geno_file = tmp_path / "genotypes.csv"
    geno_df.to_csv(geno_file, index=False)

    # Create genetic map file
    map_data = {
        'SNP': marker_names,
        'CHROM': [f"Chr{(i % 3) + 1:02d}" for i in range(n_markers)],
        'POS': [(i * 10000) + int(rng.integers(0, 5000)) for i in range(n_markers)]
    }
    map_df = pd.DataFrame(map_data)

    map_file = tmp_path / "genetic_map.csv"
    map_df.to_csv(map_file, index=False)

    # Create covariates file (numeric covariates)
    cov_df = pd.DataFrame({
        'ID': sample_ids,
        'Field': [(i % 3) for i in range(n_samples)],  # Numeric: 0, 1, 2
        'Year': [2023] * n_samples,
        'Block': rng.integers(1, 5, n_samples)  # Random blocks 1-4
    })

    cov_file = tmp_path / "covariates.csv"
    cov_df.to_csv(cov_file, index=False)

    return {
        'phenotype_file': pheno_file,
        'genotype_file': geno_file,
        'map_file': map_file,
        'covariate_file': cov_file,
        'n_samples': n_samples,
        'n_markers': n_markers,
        'sample_ids': sample_ids,
        'trait_names': ['Height', 'Yield']
    }


def test_gwas_pipeline_basic_workflow_glm(synthetic_data, tmp_path):
    """End-to-end test of GWASPipeline with GLM method."""

    output_dir = tmp_path / "gwas_results_glm"

    # Initialize pipeline
    pipeline = GWASPipeline(output_dir=str(output_dir))

    # Load data
    pipeline.load_data(
        phenotype_file=str(synthetic_data['phenotype_file']),
        genotype_file=str(synthetic_data['genotype_file']),
        map_file=str(synthetic_data['map_file']),
        trait_columns=['Height', 'Yield'],
        genotype_format='csv'
    )

    # Verify data loaded
    assert pipeline.phenotype_df is not None
    assert pipeline.genotype_matrix is not None
    assert len(pipeline.phenotype_df) == synthetic_data['n_samples']
    assert pipeline.genotype_matrix.n_markers == synthetic_data['n_markers']

    # Align samples
    pipeline.align_samples()

    # Verify alignment
    assert pipeline.genotype_matrix is not None
    assert pipeline.phenotype_df is not None

    # Run GLM analysis (no population structure needed)
    pipeline.run_analysis(
        traits=['Height'],
        methods=['GLM'],
        outputs=['all_marker_pvalues']
    )

    # Verify output files exist
    results_file = output_dir / "GWAS_Height_all_results.csv"
    assert results_file.exists(), f"Results file not found: {results_file}"

    # Load and verify results format
    results_df = pd.read_csv(results_file)

    # Check expected columns
    expected_cols = ['SNP', 'CHROM', 'POS', 'GLM_P', 'GLM_Effect']
    for col in expected_cols:
        assert col in results_df.columns, f"Missing column: {col}"

    # Check number of markers
    assert len(results_df) == synthetic_data['n_markers']

    # Check p-values are valid
    assert results_df['GLM_P'].min() >= 0.0
    assert results_df['GLM_P'].max() <= 1.0
    assert not results_df['GLM_P'].isna().any()

    # Check effects are numeric
    assert results_df['GLM_Effect'].dtype in [np.float64, np.float32, float]


def test_gwas_pipeline_mlm_with_structure(synthetic_data, tmp_path):
    """End-to-end test of GWASPipeline with MLM and population structure."""

    output_dir = tmp_path / "gwas_results_mlm"

    # Initialize pipeline
    pipeline = GWASPipeline(output_dir=str(output_dir))

    # Load data
    pipeline.load_data(
        phenotype_file=str(synthetic_data['phenotype_file']),
        genotype_file=str(synthetic_data['genotype_file']),
        map_file=str(synthetic_data['map_file']),
        trait_columns=['Height'],
        genotype_format='csv'
    )

    # Align samples
    pipeline.align_samples()

    # Compute population structure
    pipeline.compute_population_structure(
        n_pcs=3,
        calculate_kinship=True
    )

    # Verify population structure computed
    assert pipeline.pcs is not None
    assert pipeline.pcs.shape[1] == 3  # 3 PCs
    assert pipeline.kinship is not None
    assert pipeline.kinship.shape[0] == pipeline.kinship.shape[1]  # Square matrix

    # Run MLM analysis
    pipeline.run_analysis(
        traits=['Height'],
        methods=['MLM'],
        outputs=['all_marker_pvalues']
    )

    # Verify output files exist
    results_file = output_dir / "GWAS_Height_all_results.csv"
    assert results_file.exists()

    # Load and verify results
    results_df = pd.read_csv(results_file)

    # Check MLM-specific columns
    assert 'MLM_P' in results_df.columns
    assert 'MLM_Effect' in results_df.columns

    # Verify results
    assert len(results_df) == synthetic_data['n_markers']
    assert results_df['MLM_P'].min() >= 0.0
    assert results_df['MLM_P'].max() <= 1.0
    assert not results_df['MLM_P'].isna().any()


def test_gwas_pipeline_multiple_methods(synthetic_data, tmp_path):
    """Test running multiple GWAS methods in parallel."""

    output_dir = tmp_path / "gwas_results_multi"

    pipeline = GWASPipeline(output_dir=str(output_dir))

    pipeline.load_data(
        phenotype_file=str(synthetic_data['phenotype_file']),
        genotype_file=str(synthetic_data['genotype_file']),
        map_file=str(synthetic_data['map_file']),
        trait_columns=['Height'],
        genotype_format='csv'
    )

    pipeline.align_samples()
    pipeline.compute_population_structure(n_pcs=2, calculate_kinship=True)

    # Run both GLM and MLM
    pipeline.run_analysis(
        traits=['Height'],
        methods=['GLM', 'MLM'],
        outputs=['all_marker_pvalues']
    )

    # Verify results file contains both methods
    results_file = output_dir / "GWAS_Height_all_results.csv"
    results_df = pd.read_csv(results_file)

    assert 'GLM_P' in results_df.columns
    assert 'MLM_P' in results_df.columns
    assert 'GLM_Effect' in results_df.columns
    assert 'MLM_Effect' in results_df.columns


def test_gwas_pipeline_with_covariates(synthetic_data, tmp_path):
    """Test GWASPipeline with external covariates."""

    output_dir = tmp_path / "gwas_results_cov"

    pipeline = GWASPipeline(output_dir=str(output_dir))

    # Load data with covariates
    pipeline.load_data(
        phenotype_file=str(synthetic_data['phenotype_file']),
        genotype_file=str(synthetic_data['genotype_file']),
        map_file=str(synthetic_data['map_file']),
        covariate_file=str(synthetic_data['covariate_file']),
        covariate_columns=['Field'],  # Use Field as covariate
        trait_columns=['Height'],
        genotype_format='csv'
    )

    # Verify covariates loaded
    assert pipeline.covariate_df is not None
    assert 'Field' in pipeline.covariate_df.columns

    pipeline.align_samples()

    # Compute PCs - they should be combined with external covariates
    pipeline.compute_population_structure(n_pcs=2, calculate_kinship=True)

    # Run analysis
    pipeline.run_analysis(
        traits=['Height'],
        methods=['MLM'],
        outputs=['all_marker_pvalues']
    )

    # Verify results
    results_file = output_dir / "GWAS_Height_all_results.csv"
    assert results_file.exists()

    results_df = pd.read_csv(results_file)
    assert 'MLM_P' in results_df.columns
    assert not results_df['MLM_P'].isna().any()


def test_gwas_pipeline_mlm_with_lrt_refinement(synthetic_data, tmp_path):
    """Test MLM with LRT refinement (previously Hybrid MLM)."""

    output_dir = tmp_path / "gwas_results_mlm_lrt"

    pipeline = GWASPipeline(output_dir=str(output_dir))

    pipeline.load_data(
        phenotype_file=str(synthetic_data['phenotype_file']),
        genotype_file=str(synthetic_data['genotype_file']),
        map_file=str(synthetic_data['map_file']),
        trait_columns=['Height'],
        genotype_format='csv'
    )

    pipeline.align_samples()
    pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)

    # Run MLM (which now includes LRT refinement by default)
    pipeline.run_analysis(
        traits=['Height'],
        methods=['MLM'],
        outputs=['all_marker_pvalues']
    )

    # Verify output
    results_file = output_dir / "GWAS_Height_all_results.csv"
    assert results_file.exists()

    results_df = pd.read_csv(results_file)

    # MLM should have standard MLM_P column
    assert 'MLM_P' in results_df.columns

    # Verify p-values are valid
    assert results_df['MLM_P'].min() >= 0.0
    assert results_df['MLM_P'].max() <= 1.0


def test_gwas_pipeline_multiple_traits(synthetic_data, tmp_path):
    """Test analyzing multiple traits."""

    output_dir = tmp_path / "gwas_results_traits"

    pipeline = GWASPipeline(output_dir=str(output_dir))

    pipeline.load_data(
        phenotype_file=str(synthetic_data['phenotype_file']),
        genotype_file=str(synthetic_data['genotype_file']),
        map_file=str(synthetic_data['map_file']),
        trait_columns=['Height', 'Yield'],
        genotype_format='csv'
    )

    pipeline.align_samples()

    # Run GLM for both traits
    pipeline.run_analysis(
        traits=['Height', 'Yield'],
        methods=['GLM'],
        outputs=['all_marker_pvalues']
    )

    # Verify both trait results exist
    height_results = output_dir / "GWAS_Height_all_results.csv"
    yield_results = output_dir / "GWAS_Yield_all_results.csv"

    assert height_results.exists()
    assert yield_results.exists()

    # Check both have valid data
    for results_file in [height_results, yield_results]:
        df = pd.read_csv(results_file)
        assert 'GLM_P' in df.columns
        assert len(df) == synthetic_data['n_markers']


def test_gwas_pipeline_runs_without_kinship_for_loco(synthetic_data, tmp_path):
    """Test that MLM runs without precomputed kinship when LOCO is used."""

    output_dir = tmp_path / "gwas_error_test"

    pipeline = GWASPipeline(output_dir=str(output_dir))

    pipeline.load_data(
        phenotype_file=str(synthetic_data['phenotype_file']),
        genotype_file=str(synthetic_data['genotype_file']),
        map_file=str(synthetic_data['map_file']),
        trait_columns=['Height'],
        genotype_format='csv'
    )

    pipeline.align_samples()

    # Run MLM without computing kinship (LOCO does not require it)
    pipeline.run_analysis(
        traits=['Height'],
        methods=['MLM'],
        outputs=['all_marker_pvalues']
    )

    # Results should exist and include MLM output
    results_file = output_dir / "GWAS_Height_all_results.csv"
    assert results_file.exists()
    results_df = pd.read_csv(results_file)
    assert 'MLM_P' in results_df.columns
