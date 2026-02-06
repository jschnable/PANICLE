"""
File I/O utilities for PANICLE package
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Optional
import h5py

def read_phenotype(file_path: Union[str, Path]) -> pd.DataFrame:
    """Read phenotype file in rMVP format
    
    Expected format: CSV or tab-delimited, first column = individual ID
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Phenotype file not found: {file_path}")
    
    # Try to read with pandas, handling different formats
    try:
        # First try CSV with header
        df = pd.read_csv(file_path, header=0)
        if df.shape[1] < 2:
            # Try tab-separated
            df = pd.read_csv(file_path, sep='\t', header=0)
            if df.shape[1] < 2:
                # Try space-separated
                df = pd.read_csv(file_path, sep=' ', header=0)
    except:
        # Try without header
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < 2:
            df = pd.read_csv(file_path, sep='\t', header=None)
            if df.shape[1] < 2:
                df = pd.read_csv(file_path, sep=' ', header=None)
    
    if df.shape[1] < 2:
        raise ValueError(f"Phenotype file must have at least 2 columns, got {df.shape[1]}")
    
    # Take first two columns
    result = df.iloc[:, :2].copy()
    result.columns = ['ID', 'Trait']
    
    return result


def read_genotype_map(file_path: Union[str, Path]) -> pd.DataFrame:
    """Read SNP map file in rMVP format
    
    Expected columns: [SNP, Chr, Pos] minimum
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Map file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Validate required columns
    required = ['SNP', 'CHROM', 'POS']
    missing = [col for col in required if col not in df.columns]
    if missing:
        # Try alternative names
        alt_names = {'SNP': ['snp', 'marker', 'rs'], 
                     'CHROM': ['chr', 'chromosome', 'CHR'],
                     'POS': ['pos', 'position', 'bp']}
        
        for req_col in missing:
            found = False
            for alt in alt_names.get(req_col, []):
                if alt in df.columns:
                    df = df.rename(columns={alt: req_col})
                    found = True
                    break
            if not found:
                raise ValueError(f"Required column {req_col} not found in map file")
    
    return df


def write_binary_genotype(genotype: np.ndarray, output_prefix: str, 
                         sample_ids: Optional[list] = None,
                         compression: str = 'gzip') -> Tuple[str, str]:
    """Write genotype matrix in binary format for memory mapping
    
    Returns paths to genotype file and description file
    """
    output_prefix = Path(output_prefix)
    
    # Write binary genotype data
    geno_file = f"{output_prefix}.geno.bin"
    
    if compression == 'hdf5':
        # Use HDF5 for compression and fast access
        with h5py.File(f"{output_prefix}.geno.h5", 'w') as f:
            f.create_dataset('genotype', data=genotype, 
                           compression='gzip', compression_opts=9,
                           chunks=True)
    else:
        # Use numpy binary format
        genotype.astype(np.int8).tofile(geno_file)
    
    # Write description file
    desc_file = f"{output_prefix}.geno.desc"
    with open(desc_file, 'w') as f:
        f.write(f"shape: {genotype.shape[0]} {genotype.shape[1]}\n")
        f.write(f"dtype: int8\n")
        f.write(f"format: binary\n")
        if compression:
            f.write(f"compression: {compression}\n")
    
    # Write individual IDs if provided
    if sample_ids:
        ind_file = f"{output_prefix}.geno.ind"
        with open(ind_file, 'w') as f:
            for sample_id in sample_ids:
                f.write(f"{sample_id}\n")
    
    return geno_file, desc_file


def read_binary_genotype(file_prefix: str) -> Tuple[np.ndarray, dict]:
    """Read binary genotype matrix with metadata
    
    Returns genotype matrix and metadata dictionary
    """
    file_prefix = Path(file_prefix)
    
    # Read description file
    desc_file = f"{file_prefix}.geno.desc"
    if not Path(desc_file).exists():
        raise FileNotFoundError(f"Description file not found: {desc_file}")
    
    metadata = {}
    with open(desc_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                metadata[key.strip()] = value.strip()
    
    # Parse shape
    shape_str = metadata.get('shape', '')
    if not shape_str:
        raise ValueError("Shape information missing from description file")
    
    shape = tuple(map(int, shape_str.split()))
    dtype = metadata.get('dtype', 'int8')
    
    # Read genotype data
    if metadata.get('compression') == 'hdf5':
        geno_file = f"{file_prefix}.geno.h5"
        with h5py.File(geno_file, 'r') as f:
            genotype = f['genotype'][:]
    else:
        geno_file = f"{file_prefix}.geno.bin"
        genotype = np.fromfile(geno_file, dtype=dtype).reshape(shape)
    
    return genotype, metadata


def create_memmap_genotype(file_prefix: str) -> np.memmap:
    """Create memory-mapped access to binary genotype file"""
    file_prefix = Path(file_prefix)
    
    # Read metadata
    _, metadata = read_binary_genotype(file_prefix)
    
    shape_str = metadata.get('shape', '')
    shape = tuple(map(int, shape_str.split()))
    dtype = metadata.get('dtype', 'int8')
    
    if metadata.get('compression') == 'hdf5':
        raise ValueError("Cannot create memmap for compressed HDF5 files")
    
    geno_file = f"{file_prefix}.geno.bin"
    return np.memmap(geno_file, dtype=dtype, mode='r', shape=shape)


def save_association_results(results_dict: dict, output_prefix: str):
    """Save GWAS association results to files
    
    Args:
        results_dict: Dictionary with keys like 'glm', 'mlm', 'farmcpu'
        output_prefix: Output file prefix
    """
    output_prefix = Path(output_prefix)
    
    for method, results in results_dict.items():
        if results is not None:
            output_file = f"{output_prefix}.{method}.assoc.txt"
            
            # Convert to DataFrame if needed
            if hasattr(results, 'to_dataframe'):
                df = results.to_dataframe()
            elif isinstance(results, np.ndarray):
                df = pd.DataFrame(results, columns=['Effect', 'SE', 'P-value'])
            else:
                df = results
            
            # Save to file
            df.to_csv(output_file, sep='\t', index=False)


def load_association_results(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load GWAS association results from file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    return pd.read_csv(file_path, sep='\t')


def validate_input_files(phe_file: Optional[str] = None,
                        geno_file: Optional[str] = None, 
                        map_file: Optional[str] = None) -> dict:
    """Validate input files exist and have correct format
    
    Returns dictionary with validation results
    """
    results = {'valid': True, 'errors': []}
    
    if phe_file:
        phe_path = Path(phe_file)
        if not phe_path.exists():
            results['valid'] = False
            results['errors'].append(f"Phenotype file not found: {phe_file}")
        else:
            try:
                df = read_phenotype(phe_path)
                if df.shape[1] < 2:
                    results['errors'].append("Phenotype file must have at least 2 columns")
                    results['valid'] = False
            except Exception as e:
                results['errors'].append(f"Error reading phenotype file: {e}")
                results['valid'] = False
    
    if geno_file:
        geno_path = Path(geno_file)
        if not geno_path.exists():
            results['valid'] = False
            results['errors'].append(f"Genotype file not found: {geno_file}")
    
    if map_file:
        map_path = Path(map_file)
        if not map_path.exists():
            results['valid'] = False
            results['errors'].append(f"Map file not found: {map_file}")
        else:
            try:
                df = read_genotype_map(map_path)
                required = ['SNP', 'CHROM', 'POS']
                missing = [col for col in required if col not in df.columns]
                if missing:
                    results['errors'].append(f"Map file missing columns: {missing}")
                    results['valid'] = False
            except Exception as e:
                results['errors'].append(f"Error reading map file: {e}")
                results['valid'] = False
    
    return results