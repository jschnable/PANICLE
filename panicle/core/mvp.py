"""
Main MVP function - Primary GWAS analysis interface
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Any, Tuple
from pathlib import Path
import warnings

from ..utils.data_types import Phenotype, GenotypeMatrix, GenotypeMap, KinshipMatrix, AssociationResults
from ..data.loaders import load_genotype_file
from ..association.glm import PANICLE_GLM
from ..association.mlm import PANICLE_MLM
from ..association.mlm_loco import PANICLE_MLM_LOCO
from ..association.farmcpu import PANICLE_FarmCPU
from ..association.blink import PANICLE_BLINK
from ..association.farmcpu_resampling import PANICLE_FarmCPUResampling
from ..matrix.kinship import PANICLE_K_VanRaden
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
        phe: Phenotype data (file path, array, or Phenotype object)
        geno: Genotype data (file path, array, or GenotypeMatrix object)
        map_data: Genetic map data (file path, DataFrame, or GenotypeMap object)
        K: Kinship matrix (optional, calculated if not provided for MLM/FarmCPU)
        CV: Covariate matrix (optional)
        method: GWAS methods to run ["GLM", "MLM", "FarmCPU", "BLINK", "FarmCPUResampling"]
        ncpus: Number of CPU cores to use
        vc_method: Variance component method for MLM ["BRENT", "EMMA", "HE"]
        maxLine: Batch size for marker processing
        priority: Analysis priority ["speed", "memory", "accuracy"]
        threshold: Genome-wide significance threshold
        file_output: Whether to save results to files
        output_prefix: Prefix for output files
        verbose: Print progress information
        **kwargs: Additional parameters for specific methods
    
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
        
        # Load genotype data
        if isinstance(geno, (str, Path)):
            genotype, _individual_ids, loaded_map = load_genotype_file(geno)
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
        
        # Validate data consistency
        validate_data_consistency(phenotype, genotype, genetic_map, verbose)
        
        # Store processed data
        analysis_results['data'] = {
            'phenotype': phenotype,
            'genotype': genotype,
            'map': genetic_map,
            'covariates': CV
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
        
        if any(method_name in ["MLM", "FarmCPU"] for method_name in method) and K is None:
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
        
        phenotype_array = phenotype.to_numpy()
        
        # Run GLM
        if "GLM" in method:
            if verbose:
                print("\nRunning GLM analysis...")
            
            glm_start = time.time()
            glm_results = PANICLE_GLM(
                phe=phenotype_array,
                geno=genotype,
                CV=CV,
                maxLine=maxLine,
                cpu=ncpus,
                verbose=verbose
            )
            glm_time = time.time() - glm_start
            
            analysis_results['results']['GLM'] = glm_results
            analysis_results['summary']['methods_run'].append('GLM')
            analysis_results['summary']['runtime']['GLM'] = glm_time
            
            # Count significant markers
            glm_pvals = glm_results.to_numpy()[:, 2]
            n_sig = np.sum(glm_pvals < threshold)
            analysis_results['summary']['significant_markers']['GLM'] = n_sig
            
            if verbose:
                print(f"GLM analysis complete ({glm_time:.2f}s)")
                print(f"  Significant markers (p < {threshold}): {n_sig}")
        
        # Run MLM
        if "MLM" in method:
            if verbose:
                print("\nRunning MLM analysis...")

            mlm_start = time.time()
            if K is not None and verbose:
                warnings.warn("Provided kinship matrix is ignored; MLM now uses LOCO kinship.")
            mlm_results = PANICLE_MLM_LOCO(
                phe=phenotype_array,
                geno=genotype,
                map_data=genetic_map,
                CV=CV,
                vc_method=vc_method,
                maxLine=maxLine,
                cpu=ncpus,
                verbose=verbose
            )
            mlm_time = time.time() - mlm_start

            analysis_results['results']['MLM'] = mlm_results
            analysis_results['summary']['methods_run'].append('MLM')
            analysis_results['summary']['runtime']['MLM'] = mlm_time

            # Count significant markers
            mlm_pvals = mlm_results.to_numpy()[:, 2]
            n_sig = np.sum(mlm_pvals < threshold)
            analysis_results['summary']['significant_markers']['MLM'] = n_sig

            if verbose:
                print(f"MLM analysis complete ({mlm_time:.2f}s)")
                print(f"  Significant markers (p < {threshold}): {n_sig}")
        
        # Run FarmCPU
        if "FarmCPU" in method:
            if verbose:
                print("\nRunning FarmCPU analysis...")

            farmcpu_start = time.time()
            farmcpu_results = PANICLE_FarmCPU(
                phe=phenotype_array,
                geno=genotype,
                map_data=genetic_map,
                CV=CV,
                maxLine=maxLine,
                cpu=ncpus,
                verbose=verbose,
                **farmcpu_extra_kwargs
            )
            farmcpu_time = time.time() - farmcpu_start
            
            analysis_results['results']['FarmCPU'] = farmcpu_results
            analysis_results['summary']['methods_run'].append('FarmCPU')
            analysis_results['summary']['runtime']['FarmCPU'] = farmcpu_time
            
            # Count significant markers
            farmcpu_pvals = farmcpu_results.to_numpy()[:, 2]
            n_sig = np.sum(farmcpu_pvals < threshold)
            analysis_results['summary']['significant_markers']['FarmCPU'] = n_sig
            
            if verbose:
                print(f"FarmCPU analysis complete ({farmcpu_time:.2f}s)")
                print(f"  Significant markers (p < {threshold}): {n_sig}")

        # Run BLINK
        if "BLINK" in method:
            if verbose:
                print("\nRunning BLINK analysis...")

            blink_start = time.time()
            blink_results = PANICLE_BLINK(
                phe=phenotype_array,
                geno=genotype,
                map_data=genetic_map,
                CV=CV,
                maxLine=maxLine,
                cpu=ncpus,
                verbose=verbose,
                **blink_kwargs,
            )
            blink_time = time.time() - blink_start

            analysis_results['results']['BLINK'] = blink_results
            analysis_results['summary']['methods_run'].append('BLINK')
            analysis_results['summary']['runtime']['BLINK'] = blink_time

            blink_pvals = blink_results.to_numpy()[:, 2]
            n_sig = np.sum(blink_pvals < threshold)
            analysis_results['summary']['significant_markers']['BLINK'] = int(n_sig)

            if verbose:
                print(f"BLINK analysis complete ({blink_time:.2f}s)")
                print(f"  Significant markers (p < {threshold}): {n_sig}")

        # Run FarmCPU Resampling
        if "FarmCPUResampling" in method:
            if verbose:
                print("\nRunning FarmCPU resampling analysis...")

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
                geno=genotype,
                map_data=genetic_map,
                CV=CV,
                maxLine=maxLine,
                cpu=ncpus,
                trait_name=phenotype.data.columns[1] if hasattr(phenotype, 'data') else 'Trait',
                verbose=verbose,
                **resampling_params,
                **farmcpu_extra_kwargs,
            )
            resampling_time = time.time() - resampling_start

            analysis_results['results']['FarmCPUResampling'] = resampling_results
            analysis_results['summary']['methods_run'].append('FarmCPUResampling')
            analysis_results['summary']['runtime']['FarmCPUResampling'] = resampling_time

            n_identified = len(resampling_results.entries)
            analysis_results['summary']['significant_markers']['FarmCPUResampling'] = n_identified

            if verbose:
                print(f"FarmCPU resampling complete ({resampling_time:.2f}s)")
                print(f"  Markers/clusters with RMIP > 0: {n_identified}")
        
        # Phase 4: Visualization and Reporting
        if verbose:
            print(f"\n[Phase 4] Generating visualization report...")
        
        viz_start = time.time()
        visualization_report = PANICLE_Report(
            results=analysis_results['results'],
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


def save_results_to_files(results: Dict[str, Any], 
                         output_prefix: str,
                         verbose: bool = True) -> List[str]:
    """Save analysis results to files"""
    
    saved_files = []
    
    try:
        # Save summary statistics
        summary_file = f"{output_prefix}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PANICLE GWAS Analysis Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Methods run: {', '.join(results['summary']['methods_run'])}\n")
            f.write(f"Total individuals: {results['summary']['total_individuals']}\n")
            f.write(f"Total markers: {results['summary']['total_markers']}\n")
            f.write("\nSignificant markers by method:\n")
            for method, count in results['summary']['significant_markers'].items():
                f.write(f"  {method}: {count}\n")
            f.write("\nRuntimes (seconds):\n")
            for phase, time_val in results['summary']['runtime'].items():
                f.write(f"  {phase}: {time_val:.2f}s\n")
        
        saved_files.append(summary_file)
        
        # Save association results as CSV files
        for method_name, result_obj in results['results'].items():
            result_file = f"{output_prefix}_{method_name}_results.csv"
            result_df = result_obj.to_dataframe()

            # Add map information if available
            if 'map' in results['data']:
                map_obj = results['data']['map']
                if hasattr(map_obj, 'to_dataframe'):
                    map_df = map_obj.to_dataframe()
                elif hasattr(map_obj, 'data'):
                    map_df = map_obj.data
                else:
                    map_df = None

                if map_df is not None:
                    if 'SNP' not in result_df.columns:
                        result_df['SNP'] = map_df['SNP'].values[:len(result_df)]
                    if 'Chr' not in result_df.columns and 'CHROM' in map_df.columns:
                        result_df['Chr'] = map_df['CHROM'].values[:len(result_df)]
                    if 'Pos' not in result_df.columns and 'POS' in map_df.columns:
                        result_df['Pos'] = map_df['POS'].values[:len(result_df)]

            result_df.to_csv(result_file, index=False)
            saved_files.append(result_file)
        
        if verbose:
            print(f"Saved {len(saved_files)} result files")
        
    except Exception as e:
        warnings.warn(f"Failed to save some results files: {e}")
    
    return saved_files
