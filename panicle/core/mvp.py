"""
Main MVP function - Primary GWAS analysis interface
"""

import numpy as np
import pandas as pd
import json
from typing import Optional, List, Dict, Union, Any, Tuple
from pathlib import Path
import warnings

from ..utils.data_types import (
    MARKER_ID_COLUMN,
    LEGACY_MARKER_ID_COLUMN,
    infer_marker_id_column,
    Phenotype,
    GenotypeMatrix,
    GenotypeMap,
    KinshipMatrix,
    AssociationResults,
)
from ..data.loaders import load_genotype_file
from ..association.glm import PANICLE_GLM
from ..association.mlm import PANICLE_MLM
from ..association.mlm_loco import PANICLE_MLM_LOCO
from ..association.bayes_loco import PANICLE_BayesLOCO
from ..association.farmcpu import PANICLE_FarmCPU
from ..association.blink import PANICLE_BLINK
from ..association.farmcpu_resampling import PANICLE_FarmCPUResampling
from ..matrix.kinship import PANICLE_K_VanRaden
from ..matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from ..matrix.pca import PANICLE_PCA
from ..visualization.manhattan import PANICLE_Report

def PANICLE(phe: Union[str, Path, np.ndarray, pd.DataFrame, Phenotype],
        geno: Union[str, Path, np.ndarray, GenotypeMatrix],
        map_data: Union[str, Path, pd.DataFrame, GenotypeMap],
        K: Optional[Union[KinshipMatrix, np.ndarray]] = None,
        CV: Optional[np.ndarray] = None,
        method: Optional[List[str]] = None,
        ncpus: int = 1,
        vc_method: str = "BRENT",
        maxLine: int = 5000,
        priority: str = "speed",
        threshold: float = 5e-8,
        file_output: bool = True,
        output_prefix: str = "PANICLE",
        verbose: bool = True,
        **kwargs) -> Dict[str, Any]:
    """Primary GWAS analysis function
    
    Comprehensive genome-wide association study analysis supporting multiple
    statistical methods (GLM, MLM, FarmCPU) with integrated data processing,
    association testing, and visualization.
    
    Args:
        phe: Phenotype data. Accepts a file path (CSV/TSV with ID + trait columns),
            an (n, 2) numpy array where column 0 is individual ID and column 1 is
            the trait value, a DataFrame, or a Phenotype object
        geno: Genotype data (file path, array, or GenotypeMatrix object)
        map_data: Genetic map data (file path, DataFrame, or GenotypeMap object)
        K: Kinship matrix (optional, calculated if not provided for MLM/FarmCPU)
        CV: Covariate matrix (optional)
        method: GWAS methods to run ["GLM", "MLM", "BAYESLOCO", "FarmCPU", "BLINK", "FarmCPUResampling"]
        ncpus: Number of CPU cores to use
        vc_method: Variance component method for MLM ["BRENT", "EMMA", "HE"]
        maxLine: Batch size for marker processing
        priority: Analysis priority ["speed", "memory", "accuracy"]
        threshold: Genome-wide significance threshold
        file_output: Whether to save results to files
        output_prefix: Prefix for output files
        verbose: Print progress information
        **kwargs: Additional parameters for specific methods

    Notes:
        - When genotype sample IDs are available (e.g., file loaders), PANICLE
          automatically matches phenotype IDs to genotype IDs and subsets both
          datasets to their intersection.
        - For each trait, individuals with missing/non-finite phenotype values
          (or covariate values, when provided) are excluded before model fitting.
    
    Returns:
        Dictionary containing:
        - 'data': Processed input data objects
        - 'results': Association results for each method
        - 'visualization': Plots and summary statistics
        - 'files': List of created output files
    """
    
    if method is None:
        method = ["GLM"]

    if verbose:
        print("=" * 60)
        print("PANICLE: Python Algorithms for Nucleotide-phenotype")
        print("         Inference and Chromosome-wide Locus Evaluation")
        print("=" * 60)
    
    # Initialize results structure
    analysis_results = {
        'data': {},
        'results': {},
        'visualization': {},
        'files': [],
        'summary': {
            'methods_run': [],
            'total_markers': 0,
            'total_individuals': 0,
            'significant_markers': {},
            'trait_sample_sizes': {},
            'runtime': {}
        }
    }
    
    import time
    start_time = time.time()
    
    try:
        # Phase 1: Data Loading and Validation
        if verbose:
            print("\n[Phase 1] Loading and validating input data...")
        
        data_load_start = time.time()
        
        # Load phenotype data
        if isinstance(phe, (str, Path)):
            phenotype = Phenotype(phe)
        elif isinstance(phe, np.ndarray):
            phenotype = Phenotype(phe)
        elif isinstance(phe, pd.DataFrame):
            phenotype = Phenotype(phe)
        elif isinstance(phe, Phenotype):
            phenotype = phe
        else:
            raise ValueError("Invalid phenotype input type")
        
        # Normalize covariate input early so row checks and subsetting are consistent.
        covariates = None
        if CV is not None:
            covariates = np.asarray(CV)
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            if covariates.ndim != 2:
                raise ValueError("Covariate matrix must be 1D or 2D array-like")
            try:
                covariates = covariates.astype(np.float64, copy=False)
            except (TypeError, ValueError):
                raise ValueError("Covariate matrix must contain numeric values")
            if covariates.shape[0] != phenotype.n_individuals:
                raise ValueError(
                    "Covariate matrix must have the same number of rows as phenotype data"
                )

        # Load genotype data
        genotype_ids = None
        if isinstance(geno, (str, Path)):
            genotype, genotype_ids, loaded_map = load_genotype_file(geno)
            # Use the map embedded in the genotype file when map_data is not
            # explicitly provided (i.e. caller passed the same path or None).
            if isinstance(map_data, (str, Path)) or map_data is None:
                map_data = loaded_map
        elif isinstance(geno, np.ndarray):
            genotype = GenotypeMatrix(geno)
        elif isinstance(geno, GenotypeMatrix):
            genotype = geno
        else:
            raise ValueError("Invalid genotype input type")

        # Load map data
        if isinstance(map_data, (str, Path)):
            genetic_map = GenotypeMap(map_data)
        elif isinstance(map_data, pd.DataFrame):
            genetic_map = GenotypeMap(map_data)
        elif isinstance(map_data, GenotypeMap):
            genetic_map = map_data
        else:
            raise ValueError("Invalid map input type")

        # Automatically align phenotype/covariates to genotype IDs when available.
        if genotype_ids is not None:
            phenotype, genotype, covariates, matching_summary = _align_samples_to_genotype(
                phenotype=phenotype,
                genotype=genotype,
                genotype_ids=genotype_ids,
                covariates=covariates,
            )
            analysis_results['summary']['sample_matching'] = matching_summary
            if verbose:
                print("Sample matching complete")
                print(f"  Matched individuals: {matching_summary['n_common']}")
                print(f"  Dropped phenotype-only IDs: {matching_summary['n_phenotype_dropped']}")
                print(f"  Dropped genotype-only IDs: {matching_summary['n_genotype_dropped']}")
        elif phenotype.n_individuals != genotype.n_individuals:
            raise ValueError(
                "Phenotype and genotype have different numbers of individuals, "
                "and genotype sample IDs are unavailable for automatic alignment."
            )
        
        # Validate data consistency
        validate_data_consistency(phenotype, genotype, genetic_map, verbose)
        
        # Store processed data
        analysis_results['data'] = {
            'phenotype': phenotype,
            'genotype': genotype,
            'map': genetic_map,
            'covariates': covariates
        }
        
        analysis_results['summary']['total_markers'] = genotype.n_markers
        analysis_results['summary']['total_individuals'] = genotype.n_individuals
        
        data_load_time = time.time() - data_load_start
        analysis_results['summary']['runtime']['data_loading'] = data_load_time
        
        if verbose:
            print(f"Data loading complete ({data_load_time:.2f}s)")
            print(f"  Individuals: {genotype.n_individuals}")
            print(f"  Markers: {genotype.n_markers}")
            print(f"  Traits: {phenotype.n_traits}")
        
        # Phase 2: Kinship Matrix and PCA (if needed)
        kinship_matrix = None
        pca_results = None
        
        if any(method_name in ["FarmCPU"] for method_name in method) and K is None:
            if verbose:
                print("\n[Phase 2] Computing kinship matrix...")
            
            kinship_start = time.time()
            kinship_matrix = PANICLE_K_VanRaden(
                genotype, 
                maxLine=maxLine,
                verbose=verbose
            )
            kinship_time = time.time() - kinship_start
            analysis_results['summary']['runtime']['kinship'] = kinship_time
            
            if verbose:
                print(f"Kinship matrix computation complete ({kinship_time:.2f}s)")
        
        elif K is not None:
            kinship_matrix = K
            if verbose:
                print("\n[Phase 2] Using provided kinship matrix")
        
        # Store kinship matrix
        if kinship_matrix is not None:
            analysis_results['data']['kinship'] = kinship_matrix
        
        # Extract FarmCPU resampling parameters to avoid propagating them to other methods
        resampling_significance_kwarg = kwargs.pop('farmcpu_resampling_significance_threshold', None)
        resampling_params = {
            'runs': kwargs.pop('farmcpu_resampling_runs', 100),
            'mask_proportion': kwargs.pop('farmcpu_resampling_mask_proportion', 0.1),
            'significance_threshold': threshold,
            'cluster_markers': kwargs.pop('farmcpu_resampling_cluster', False),
            'ld_threshold': kwargs.pop('farmcpu_resampling_ld_threshold', 0.7),
            'random_seed': kwargs.pop('farmcpu_resampling_random_seed', None),
        }
        resampling_override_used = resampling_significance_kwarg is not None
        if resampling_override_used:
            resampling_params['significance_threshold'] = resampling_significance_kwarg

        farmcpu_extra_keys = [
            'maxLoop', 'p_threshold', 'QTN_threshold', 'bin_size',
            'method_bin', 'reward_method'
        ]
        farmcpu_extra_kwargs = {
            key: kwargs[key] for key in farmcpu_extra_keys if key in kwargs
        }

        blink_extra_keys = [
            'maxLoop', 'converge', 'ld_threshold', 'maf_threshold',
            'bic_method', 'method_sub', 'p_threshold', 'qtn_threshold',
            'cut_off', 'fdr_cut'
        ]
        blink_kwargs = {key: kwargs[key] for key in blink_extra_keys if key in kwargs}
        blink_prior = kwargs.get('Prior', kwargs.get('prior'))
        if blink_prior is not None:
            blink_kwargs['Prior'] = blink_prior

        # Phase 3: Association Analysis
        if verbose:
            print(f"\n[Phase 3] Running association analysis...")
            print(f"Methods: {', '.join(method)}")
            print(f"Traits: {', '.join(phenotype.trait_names)}")

        # Track which methods were run (only add once, not per-trait)
        methods_actually_run = set()
        covariate_finite_mask = (
            np.isfinite(covariates).all(axis=1) if covariates is not None else None
        )
        loco_kinship_cache: Dict[Tuple[int, int], Any] = {}

        # Loop over each trait
        for trait_idx, trait_name in enumerate(phenotype.trait_names):
            if verbose and phenotype.n_traits > 1:
                print(f"\n--- Analyzing trait: {trait_name} ({trait_idx + 1}/{phenotype.n_traits}) ---")

            # Trait-specific filtering: exclude missing/non-finite phenotype and covariates.
            raw_trait = pd.to_numeric(phenotype.get_trait(trait_idx), errors='coerce').to_numpy(dtype=np.float64)
            trait_ids = phenotype.ids.astype(str).to_numpy()
            valid_mask = np.isfinite(raw_trait)
            if covariate_finite_mask is not None:
                valid_mask = valid_mask & covariate_finite_mask

            n_valid = int(valid_mask.sum())
            if n_valid == 0:
                raise ValueError(
                    f"Trait '{trait_name}' has no valid observations after excluding missing phenotype/covariate values."
                )

            if n_valid < len(valid_mask):
                excluded = len(valid_mask) - n_valid
                if verbose:
                    print(
                        f"  Trait '{trait_name}': excluded {excluded} individual(s) with missing/non-finite phenotype or covariate values."
                    )

            valid_indices = np.where(valid_mask)[0]
            phenotype_array = np.column_stack([trait_ids[valid_mask], raw_trait[valid_mask]])
            trait_genotype = genotype if n_valid == genotype.n_individuals else genotype.subset_individuals(valid_indices)
            trait_covariates = covariates[valid_mask, :] if covariates is not None else None
            analysis_results['summary']['trait_sample_sizes'][trait_name] = n_valid

            # Initialize results dict for this trait
            analysis_results['results'][trait_name] = {}
            analysis_results['summary']['significant_markers'][trait_name] = {}

            # Run GLM
            if "GLM" in method:
                if verbose:
                    print(f"\nRunning GLM analysis on {trait_name}...")

                glm_start = time.time()
                glm_results = PANICLE_GLM(
                    phe=phenotype_array,
                    geno=trait_genotype,
                    CV=trait_covariates,
                    maxLine=maxLine,
                    cpu=ncpus,
                    verbose=verbose
                )
                glm_time = time.time() - glm_start

                analysis_results['results'][trait_name]['GLM'] = glm_results
                methods_actually_run.add('GLM')
                analysis_results['summary']['runtime'][f'GLM_{trait_name}'] = glm_time

                # Count significant markers
                glm_pvals = glm_results.to_numpy()[:, 2]
                n_sig = np.sum(glm_pvals < threshold)
                analysis_results['summary']['significant_markers'][trait_name]['GLM'] = n_sig

                if verbose:
                    print(f"GLM analysis complete ({glm_time:.2f}s)")
                    print(f"  Significant markers (p < {threshold}): {n_sig}")

            # Run MLM
            if "MLM" in method:
                if verbose:
                    print(f"\nRunning MLM analysis on {trait_name}...")

                mlm_start = time.time()
                if K is not None and verbose and trait_idx == 0:
                    warnings.warn("Provided kinship matrix is ignored; MLM now uses LOCO kinship.")
                key_arr = np.ascontiguousarray(valid_indices, dtype=np.int64)
                loco_key = (int(key_arr.size), hash(key_arr.tobytes()))
                trait_loco_kinship = loco_kinship_cache.get(loco_key)
                if trait_loco_kinship is None:
                    trait_loco_kinship = PANICLE_K_VanRaden_LOCO(
                        trait_genotype,
                        genetic_map,
                        maxLine=maxLine,
                        cpu=ncpus,
                        verbose=False,
                    )
                    loco_kinship_cache[loco_key] = trait_loco_kinship
                mlm_results = PANICLE_MLM_LOCO(
                    phe=phenotype_array,
                    geno=trait_genotype,
                    map_data=genetic_map,
                    loco_kinship=trait_loco_kinship,
                    CV=trait_covariates,
                    vc_method=vc_method,
                    maxLine=maxLine,
                    cpu=ncpus,
                    verbose=verbose
                )
                mlm_time = time.time() - mlm_start

                analysis_results['results'][trait_name]['MLM'] = mlm_results
                methods_actually_run.add('MLM')
                analysis_results['summary']['runtime'][f'MLM_{trait_name}'] = mlm_time

                # Count significant markers
                mlm_pvals = mlm_results.to_numpy()[:, 2]
                n_sig = np.sum(mlm_pvals < threshold)
                analysis_results['summary']['significant_markers'][trait_name]['MLM'] = n_sig

                if verbose:
                    print(f"MLM analysis complete ({mlm_time:.2f}s)")
                    print(f"  Significant markers (p < {threshold}): {n_sig}")

            # Run BAYESLOCO
            if "BAYESLOCO" in method:
                if verbose:
                    print(f"\nRunning BAYESLOCO analysis on {trait_name}...")

                bayes_cfg = kwargs.get("bayesloco_config", kwargs.get("bl_config"))
                if bayes_cfg is None:
                    bayes_cfg = {}
                if isinstance(bayes_cfg, dict):
                    bayes_cfg = dict(bayes_cfg)
                    # Reuse PANICLE maxLine as BAYESLOCO marker batch defaults when not overridden.
                    bayes_cfg.setdefault("batch_markers_fit", int(maxLine))
                    bayes_cfg.setdefault("batch_markers_test", int(maxLine))
                bayes_start = time.time()
                bayes_results = PANICLE_BayesLOCO(
                    phe=phenotype_array,
                    geno=trait_genotype,
                    map_data=genetic_map,
                    CV=trait_covariates,
                    cpu=ncpus,
                    verbose=verbose,
                    bl_config=bayes_cfg,
                )
                bayes_time = time.time() - bayes_start

                analysis_results['results'][trait_name]['BAYESLOCO'] = bayes_results
                methods_actually_run.add('BAYESLOCO')
                analysis_results['summary']['runtime'][f'BAYESLOCO_{trait_name}'] = bayes_time

                bayes_pvals = bayes_results.to_numpy()[:, 2]
                n_sig = np.sum(bayes_pvals < threshold)
                analysis_results['summary']['significant_markers'][trait_name]['BAYESLOCO'] = int(n_sig)

                if verbose:
                    print(f"BAYESLOCO analysis complete ({bayes_time:.2f}s)")
                    print(f"  Significant markers (p < {threshold}): {n_sig}")

            # Run FarmCPU
            if "FarmCPU" in method:
                if verbose:
                    print(f"\nRunning FarmCPU analysis on {trait_name}...")

                farmcpu_start = time.time()
                farmcpu_results = PANICLE_FarmCPU(
                    phe=phenotype_array,
                    geno=trait_genotype,
                    map_data=genetic_map,
                    CV=trait_covariates,
                    maxLine=maxLine,
                    cpu=ncpus,
                    verbose=verbose,
                    **farmcpu_extra_kwargs
                )
                farmcpu_time = time.time() - farmcpu_start

                analysis_results['results'][trait_name]['FarmCPU'] = farmcpu_results
                methods_actually_run.add('FarmCPU')
                analysis_results['summary']['runtime'][f'FarmCPU_{trait_name}'] = farmcpu_time

                # Count significant markers
                farmcpu_pvals = farmcpu_results.to_numpy()[:, 2]
                n_sig = np.sum(farmcpu_pvals < threshold)
                analysis_results['summary']['significant_markers'][trait_name]['FarmCPU'] = n_sig

                if verbose:
                    print(f"FarmCPU analysis complete ({farmcpu_time:.2f}s)")
                    print(f"  Significant markers (p < {threshold}): {n_sig}")

            # Run BLINK
            if "BLINK" in method:
                if verbose:
                    print(f"\nRunning BLINK analysis on {trait_name}...")

                blink_start = time.time()
                blink_results = PANICLE_BLINK(
                    phe=phenotype_array,
                    geno=trait_genotype,
                    map_data=genetic_map,
                    CV=trait_covariates,
                    maxLine=maxLine,
                    cpu=ncpus,
                    verbose=verbose,
                    **blink_kwargs,
                )
                blink_time = time.time() - blink_start

                analysis_results['results'][trait_name]['BLINK'] = blink_results
                methods_actually_run.add('BLINK')
                analysis_results['summary']['runtime'][f'BLINK_{trait_name}'] = blink_time

                blink_pvals = blink_results.to_numpy()[:, 2]
                n_sig = np.sum(blink_pvals < threshold)
                analysis_results['summary']['significant_markers'][trait_name]['BLINK'] = int(n_sig)

                if verbose:
                    print(f"BLINK analysis complete ({blink_time:.2f}s)")
                    print(f"  Significant markers (p < {threshold}): {n_sig}")

            # Run FarmCPU Resampling
            if "FarmCPUResampling" in method:
                if verbose:
                    print(f"\nRunning FarmCPU resampling analysis on {trait_name}...")

                resampling_start = time.time()
                resampling_significance = resampling_params.get('significance_threshold')
                if resampling_significance is None:
                    resampling_significance = threshold
                    resampling_params['significance_threshold'] = resampling_significance
                resampling_p_threshold = farmcpu_extra_kwargs.get('p_threshold', 0.05)
                resampling_qtn_threshold = max(resampling_p_threshold, farmcpu_extra_kwargs.get('QTN_threshold', 0.01))
                if resampling_significance > resampling_qtn_threshold and resampling_override_used:
                    warnings.warn(
                        "FarmCPU resampling significance threshold "
                        f"({resampling_significance:.3g}) is less stringent than the "
                        f"QTN threshold ({resampling_qtn_threshold:.3g}); markers with "
                        "p-values above the QTN threshold cannot act as pseudo QTNs in "
                        "later FarmCPU iterations."
                    )
                resampling_results = PANICLE_FarmCPUResampling(
                    phe=phenotype_array,
                    geno=trait_genotype,
                    map_data=genetic_map,
                    CV=trait_covariates,
                    maxLine=maxLine,
                    cpu=ncpus,
                    trait_name=trait_name,
                    verbose=verbose,
                    **resampling_params,
                    **farmcpu_extra_kwargs,
                )
                resampling_time = time.time() - resampling_start

                analysis_results['results'][trait_name]['FarmCPUResampling'] = resampling_results
                methods_actually_run.add('FarmCPUResampling')
                analysis_results['summary']['runtime'][f'FarmCPUResampling_{trait_name}'] = resampling_time

                n_identified = len(resampling_results.entries)
                analysis_results['summary']['significant_markers'][trait_name]['FarmCPUResampling'] = n_identified

                if verbose:
                    print(f"FarmCPU resampling complete ({resampling_time:.2f}s)")
                    print(f"  Markers/clusters with RMIP > 0: {n_identified}")

        # Record which methods were run
        analysis_results['summary']['methods_run'] = list(methods_actually_run)
        analysis_results['summary']['n_traits'] = phenotype.n_traits
        analysis_results['summary']['trait_names'] = phenotype.trait_names
        
        # Phase 4: Visualization and Reporting
        if verbose:
            print(f"\n[Phase 4] Generating visualization report...")

        viz_start = time.time()

        # Flatten results for visualization: {trait_method: result_obj}
        flat_results = {}
        for trait_name, trait_results in analysis_results['results'].items():
            for method_name, result_obj in trait_results.items():
                # Use trait name in key only if multiple traits
                if phenotype.n_traits == 1:
                    key = method_name
                else:
                    key = f"{trait_name}_{method_name}"
                flat_results[key] = result_obj

        visualization_report = PANICLE_Report(
            results=flat_results,
            map_data=genetic_map,
            threshold=threshold,
            output_prefix=output_prefix,
            save_plots=file_output,
            verbose=verbose
        )
        viz_time = time.time() - viz_start
        
        analysis_results['visualization'] = visualization_report
        analysis_results['summary']['runtime']['visualization'] = viz_time
        
        if file_output:
            analysis_results['files'].extend(visualization_report['files_created'])
        
        if verbose:
            print(f"Visualization complete ({viz_time:.2f}s)")
            print(f"  Generated {len(visualization_report['files_created'])} plot files")
        
        # Phase 5: Save Results
        if file_output:
            if verbose:
                print(f"\n[Phase 5] Saving results to files...")
            
            save_start = time.time()
            saved_files = save_results_to_files(
                analysis_results, 
                output_prefix, 
                verbose
            )
            save_time = time.time() - save_start
            
            analysis_results['files'].extend(saved_files)
            analysis_results['summary']['runtime']['file_output'] = save_time
            
            if verbose:
                print(f"Results saved ({save_time:.2f}s)")
        
        # Final summary
        total_time = time.time() - start_time
        analysis_results['summary']['runtime']['total'] = total_time
        
        if verbose:
            print(f"\n" + "=" * 60)
            print("GWAS Analysis Complete!")
            print(f"Total runtime: {total_time:.2f}s")
            print(f"Methods run: {', '.join(analysis_results['summary']['methods_run'])}")
            print(f"Total files created: {len(analysis_results['files'])}")
            print("=" * 60)
        
        return analysis_results
        
    except Exception as e:
        if verbose:
            print(f"\nERROR: GWAS analysis failed: {str(e)}")
        raise


def validate_data_consistency(phenotype: Phenotype, 
                            genotype: GenotypeMatrix, 
                            genetic_map: GenotypeMap,
                            verbose: bool = True):
    """Validate consistency between phenotype, genotype, and map data"""
    
    # Check that number of markers matches
    map_length = len(genetic_map.data) if hasattr(genetic_map, 'data') else len(genetic_map)
    if genotype.n_markers != map_length:
        raise ValueError(
            f"Genotype markers ({genotype.n_markers}) does not match "
            f"map entries ({map_length})"
        )
    
    # Check for reasonable data sizes
    if genotype.n_individuals < 10:
        warnings.warn("Very few individuals (<10) for GWAS analysis")
    
    if genotype.n_markers < 100:
        warnings.warn("Very few markers (<100) for GWAS analysis")
    
    # Check for missing data rates
    if hasattr(genotype, 'calculate_missing_rate'):
        missing_rate = genotype.calculate_missing_rate()
        if missing_rate > 0.1:
            warnings.warn(f"High missing data rate: {missing_rate:.2%}")
    
    if verbose:
        print("Data consistency validation passed")


def _align_samples_to_genotype(
    phenotype: Phenotype,
    genotype: GenotypeMatrix,
    genotype_ids: List[Any],
    covariates: Optional[np.ndarray] = None,
) -> Tuple[Phenotype, GenotypeMatrix, Optional[np.ndarray], Dict[str, int]]:
    """Align phenotype (and optional covariates) to genotype sample IDs.

    Keeps phenotype row order, subsets genotype to matching rows, and reports
    how many IDs were retained/dropped.
    """
    phenotype_df = phenotype.data.copy()
    phenotype_df['ID'] = phenotype_df['ID'].astype(str)

    genotype_ids_str = [str(sample_id) for sample_id in genotype_ids]
    id_to_genotype_index = {sample_id: idx for idx, sample_id in enumerate(genotype_ids_str)}

    phe_ids = phenotype_df['ID'].to_numpy()
    keep_mask = np.array([sample_id in id_to_genotype_index for sample_id in phe_ids], dtype=bool)

    n_common = int(keep_mask.sum())
    if n_common == 0:
        raise ValueError("No common sample IDs between phenotype and genotype data")

    aligned_phenotype_df = phenotype_df.loc[keep_mask].reset_index(drop=True)
    aligned_ids = aligned_phenotype_df['ID'].tolist()
    genotype_indices = np.array([id_to_genotype_index[sample_id] for sample_id in aligned_ids], dtype=int)
    aligned_genotype = genotype.subset_individuals(genotype_indices)

    aligned_covariates = None
    if covariates is not None:
        aligned_covariates = covariates[keep_mask, :]

    unique_phe = set(phe_ids.tolist())
    unique_geno = set(genotype_ids_str)
    common_ids = unique_phe & unique_geno

    summary = {
        'n_phenotype_original': len(unique_phe),
        'n_genotype_original': len(unique_geno),
        'n_common': len(common_ids),
        'n_phenotype_dropped': len(unique_phe - common_ids),
        'n_genotype_dropped': len(unique_geno - common_ids),
    }

    return Phenotype(aligned_phenotype_df), aligned_genotype, aligned_covariates, summary


def save_results_to_files(results: Dict[str, Any],
                         output_prefix: str,
                         verbose: bool = True) -> List[str]:
    """Save analysis results to files"""

    saved_files = []

    def _json_default(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    try:
        # Save summary statistics
        summary_file = f"{output_prefix}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PANICLE GWAS Analysis Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Methods run: {', '.join(results['summary']['methods_run'])}\n")
            f.write(f"Total individuals: {results['summary']['total_individuals']}\n")
            f.write(f"Total markers: {results['summary']['total_markers']}\n")
            n_traits = results['summary'].get('n_traits', 1)
            trait_names = results['summary'].get('trait_names', ['Trait'])
            f.write(f"Traits analyzed: {n_traits} ({', '.join(trait_names)})\n")
            f.write("\nSignificant markers by trait and method:\n")
            for trait_name, methods in results['summary']['significant_markers'].items():
                f.write(f"  {trait_name}:\n")
                for method, count in methods.items():
                    f.write(f"    {method}: {count}\n")
            f.write("\nRuntimes (seconds):\n")
            for phase, time_val in results['summary']['runtime'].items():
                f.write(f"  {phase}: {time_val:.2f}s\n")

        saved_files.append(summary_file)

        # Get map data once for reuse
        map_df = None
        if 'map' in results['data']:
            map_obj = results['data']['map']
            if hasattr(map_obj, 'to_dataframe'):
                map_df = map_obj.to_dataframe()
            elif hasattr(map_obj, 'data'):
                map_df = map_obj.data

        # Save association results as CSV files (nested by trait)
        for trait_name, trait_results in results['results'].items():
            for method_name, result_obj in trait_results.items():
                result_file = f"{output_prefix}_{trait_name}_{method_name}_results.csv"
                result_df = result_obj.to_dataframe()

                # Add map information if available
                if map_df is not None:
                    marker_col = infer_marker_id_column(map_df.columns)
                    if marker_col is not None:
                        if MARKER_ID_COLUMN not in result_df.columns:
                            result_df[MARKER_ID_COLUMN] = map_df[marker_col].values[:len(result_df)]
                        if LEGACY_MARKER_ID_COLUMN not in result_df.columns:
                            result_df[LEGACY_MARKER_ID_COLUMN] = result_df[MARKER_ID_COLUMN].astype(str)
                    if 'Chr' not in result_df.columns and 'CHROM' in map_df.columns:
                        result_df['Chr'] = map_df['CHROM'].values[:len(result_df)]
                    if 'Pos' not in result_df.columns and 'POS' in map_df.columns:
                        result_df['Pos'] = map_df['POS'].values[:len(result_df)]

                result_df.to_csv(result_file, index=False)
                saved_files.append(result_file)

                metadata = getattr(result_obj, "metadata", None)
                if isinstance(metadata, dict) and metadata:
                    meta_file = f"{output_prefix}_{trait_name}_{method_name}_metadata.json"
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, sort_keys=True, default=_json_default)
                    saved_files.append(meta_file)

        if verbose:
            print(f"Saved {len(saved_files)} result files")

    except Exception as e:
        warnings.warn(f"Failed to save some results files: {e}")

    return saved_files
