"""
Memory-mapped file utilities for efficient handling of large genotype datasets
Compatible with rMVP and optimized for PANICLE FarmCPU performance
"""

import numpy as np
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union, Sequence
import pandas as pd
from .data_types import GenotypeMatrix, GenotypeMap


def save_genotype_to_memmap(genotype: Union[GenotypeMatrix, np.ndarray],
                            sample_ids: Sequence[str],
                            geno_map: Optional[GenotypeMap],
                            output_prefix: Union[str, Path],
                            dtype: np.dtype = np.int8,
                            batch_size: int = 1000) -> dict:
    """Persist genotype data to a reusable memmap with metadata.

    Args:
        genotype: GenotypeMatrix or ndarray (n_individuals × n_markers)
        sample_ids: Iterable of sample identifiers
        geno_map: Optional GenotypeMap; stored to CSV if provided
        output_prefix: Base path for emitted files (without extension)
        dtype: Storage dtype for memmap
        batch_size: Marker batch size when copying data

    Returns:
        Dictionary describing generated artefacts.
    """

    dtype = np.dtype(dtype)
    output_prefix = Path(output_prefix)
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(genotype, GenotypeMatrix):
        shape = genotype.shape
        getter = genotype.get_batch
    else:
        array = np.asarray(genotype)
        shape = array.shape

        def getter(start: int, end: int):
            return array[:, start:end]

    n_individuals, n_markers = shape
    memmap_path = output_prefix.with_suffix('.memmap')
    metadata_path = output_prefix.with_suffix('.pymvp_cache.npz')
    samples_path = output_prefix.with_suffix('.samples.txt')
    map_path = output_prefix.with_suffix('.map.csv')

    mm = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=shape)
    try:
        for start in range(0, n_markers, batch_size):
            end = min(start + batch_size, n_markers)
            batch = getter(start, end)
            if not isinstance(batch, np.ndarray):  # safety for sequences
                batch = np.asarray(batch)
            mm[:, start:end] = batch.astype(dtype, copy=False)
        mm.flush()
    finally:
        del mm

    # Persist sample IDs
    with open(samples_path, 'w') as handle:
        for sid in sample_ids:
            handle.write(f"{sid}\n")

    # Persist map if available
    map_file_written: Optional[Path] = None
    if geno_map is not None:
        map_df = geno_map.to_dataframe()
        map_df.to_csv(map_path, index=False)
        map_file_written = map_path

    np.savez_compressed(
        metadata_path,
        shape=shape,
        dtype=dtype.str,
        memmap_file=str(memmap_path.resolve()),
        samples_file=str(samples_path.resolve()),
        map_file=str(map_file_written.resolve()) if map_file_written else '',
        writer_version="1",
    )

    return {
        'memmap_path': memmap_path,
        'metadata_path': metadata_path,
        'samples_path': samples_path,
        'map_path': map_file_written,
        'shape': shape,
        'dtype': dtype,
    }


def load_full_from_metadata(metadata_path: Union[str, Path],
                            precompute_alleles: bool = True) -> Tuple[GenotypeMatrix, Sequence[str], Optional[GenotypeMap]]:
    """Load genotype, sample IDs, and map from a PANICLE cache metadata file."""

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = np.load(metadata_path, allow_pickle=True)

    shape = tuple(int(x) for x in metadata['shape'])
    dtype_value = metadata['dtype']
    if isinstance(dtype_value, np.ndarray):
        dtype_value = dtype_value.item()
    dtype = np.dtype(dtype_value)
    memmap_file = Path(metadata['memmap_file'].item())

    genotype = GenotypeMatrix(
        data=str(memmap_file),
        shape=shape,
        dtype=dtype,
        precompute_alleles=precompute_alleles,
    )

    samples_file = metadata.get('samples_file')
    if samples_file is not None:
        samples_path = Path(samples_file.item())
        with open(samples_path, 'r') as handle:
            sample_ids = [line.strip() for line in handle if line.strip()]
    else:
        sample_ids = [f"Sample_{i+1}" for i in range(shape[0])]

    map_file = metadata.get('map_file')
    geno_map: Optional[GenotypeMap]
    if map_file is not None:
        map_path_str = map_file.item()
        if map_path_str:
            map_df = pd.read_csv(map_path_str, dtype={'SNP': str, 'CHROM': str, 'REF': str, 'ALT': str})
            geno_map = GenotypeMap(map_df)
        else:
            geno_map = None
    else:
        geno_map = None

    return genotype, sample_ids, geno_map

def create_memmap_from_csv(csv_path: Union[str, Path], 
                          output_path: Optional[Union[str, Path]] = None,
                          dtype: np.dtype = np.int8,
                          chunk_size: int = 1000) -> Tuple[Path, Tuple[int, int]]:
    """Create memory-mapped file from CSV genotype data
    
    Args:
        csv_path: Path to CSV file with genotype data
        output_path: Output path for memmap file (auto-generated if None)
        dtype: Data type for the memmap array
        chunk_size: Number of rows to process at once
    
    Returns:
        Tuple of (memmap_file_path, (n_individuals, n_markers))
    """
    csv_path = Path(csv_path)
    
    if output_path is None:
        output_path = csv_path.with_suffix('.memmap')
    else:
        output_path = Path(output_path)
    
    # First pass: determine dimensions
    print(f"📊 Analyzing {csv_path} to determine dimensions...")
    
    # Read first chunk to get number of columns
    first_chunk = pd.read_csv(csv_path, nrows=chunk_size, index_col=0)
    n_markers = first_chunk.shape[1]
    
    # Count total rows efficiently
    with open(csv_path, 'r') as f:
        n_individuals = sum(1 for line in f) - 1  # Subtract header
    
    print(f"📊 Dataset dimensions: {n_individuals} individuals × {n_markers} markers")
    print(f"💾 Creating memory-mapped file: {output_path}")
    
    # Create memory-mapped array
    shape = (n_individuals, n_markers)
    memmap_array = np.memmap(output_path, dtype=dtype, mode='w+', shape=shape)
    
    # Second pass: fill the memmap array in chunks
    print(f"📊 Processing data in chunks of {chunk_size} individuals...")
    
    chunk_iter = pd.read_csv(csv_path, index_col=0, chunksize=chunk_size)
    
    start_row = 0
    for chunk_idx, chunk in enumerate(chunk_iter):
        end_row = start_row + len(chunk)
        
        if chunk_idx % 10 == 0:
            print(f"  Processing chunk {chunk_idx + 1}, rows {start_row}-{end_row-1}")
        
        # Convert chunk to appropriate dtype and store
        chunk_array = chunk.values.astype(dtype)
        memmap_array[start_row:end_row, :] = chunk_array
        
        start_row = end_row
    
    # Flush to disk
    del memmap_array
    
    print(f"✅ Memory-mapped file created: {output_path}")
    print(f"💾 File size: {output_path.stat().st_size / (1024**2):.1f} MB")
    
    return output_path, shape


def load_genotype_memmap(memmap_path: Union[str, Path],
                        shape: Tuple[int, int],
                        dtype: np.dtype = np.int8,
                        precompute_alleles: bool = True) -> GenotypeMatrix:
    """Load genotype data from memory-mapped file
    
    Args:
        memmap_path: Path to memory-mapped file
        shape: Shape of the genotype matrix (n_individuals, n_markers)
        dtype: Data type of the memmap array
        precompute_alleles: Whether to pre-compute major alleles
    
    Returns:
        GenotypeMatrix object with memory-mapped backend
    """
    memmap_path = Path(memmap_path)
    
    if not memmap_path.exists():
        raise FileNotFoundError(f"Memory-mapped file not found: {memmap_path}")
    
    print(f"📊 Loading memory-mapped genotype data from {memmap_path}")
    print(f"📊 Shape: {shape[0]} individuals × {shape[1]} markers")
    print(f"💾 File size: {memmap_path.stat().st_size / (1024**2):.1f} MB")
    
    # Create GenotypeMatrix with memory-mapped backend
    geno_matrix = GenotypeMatrix(
        data=str(memmap_path),
        shape=shape,
        dtype=dtype,
        precompute_alleles=precompute_alleles
    )
    
    print(f"✅ Memory-mapped GenotypeMatrix loaded successfully")
    
    return geno_matrix


def convert_csv_to_optimized_format(csv_path: Union[str, Path],
                                  output_dir: Optional[Union[str, Path]] = None,
                                  dtype: np.dtype = np.int8,
                                  chunk_size: int = 1000) -> dict:
    """Convert CSV genotype data to optimized memory-mapped format
    
    Creates both the memory-mapped file and metadata for efficient loading.
    
    Args:
        csv_path: Path to CSV genotype file
        output_dir: Output directory (same as input if None)
        dtype: Data type for storage
        chunk_size: Processing chunk size
    
    Returns:
        Dictionary with paths and metadata
    """
    csv_path = Path(csv_path)
    
    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output paths
    base_name = csv_path.stem
    memmap_path = output_dir / f"{base_name}.memmap"
    metadata_path = output_dir / f"{base_name}_metadata.npz"
    
    print(f"🔄 Converting {csv_path} to optimized format...")
    
    # Create memory-mapped file
    memmap_file, shape = create_memmap_from_csv(
        csv_path, memmap_path, dtype, chunk_size
    )
    
    # Create metadata file
    print(f"💾 Saving metadata to {metadata_path}")
    np.savez_compressed(
        metadata_path,
        shape=shape,
        dtype=str(dtype),
        source_file=str(csv_path),
        memmap_file=str(memmap_file)
    )
    
    result = {
        'memmap_path': memmap_file,
        'metadata_path': metadata_path,
        'shape': shape,
        'dtype': dtype,
        'size_mb': memmap_file.stat().st_size / (1024**2)
    }
    
    print(f"✅ Conversion complete!")
    print(f"   Memory-mapped file: {memmap_file}")
    print(f"   Metadata file: {metadata_path}")
    print(f"   Size: {result['size_mb']:.1f} MB")
    
    return result


def load_from_metadata(metadata_path: Union[str, Path],
                      precompute_alleles: bool = True) -> GenotypeMatrix:
    """Load GenotypeMatrix from metadata file
    
    Args:
        metadata_path: Path to metadata .npz file
        precompute_alleles: Whether to pre-compute major alleles
    
    Returns:
        GenotypeMatrix object
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load metadata
    metadata = np.load(metadata_path, allow_pickle=True)
    
    shape = tuple(metadata['shape'])
    dtype = np.dtype(metadata['dtype'].item())
    memmap_path = Path(metadata['memmap_file'].item())
    
    # Load genotype matrix
    return load_genotype_memmap(
        memmap_path, shape, dtype, precompute_alleles
    )


class OptimizedGenotypeLoader:
    """High-level interface for loading genotype data in optimal format"""
    
    def __init__(self, data_path: Union[str, Path], 
                 force_conversion: bool = False,
                 precompute_alleles: bool = True):
        """Initialize loader
        
        Args:
            data_path: Path to genotype data (CSV or metadata file)
            force_conversion: Force re-conversion even if optimized files exist
            precompute_alleles: Pre-compute major alleles for fast imputation
        """
        self.data_path = Path(data_path)
        self.force_conversion = force_conversion
        self.precompute_alleles = precompute_alleles
        
    def load(self) -> GenotypeMatrix:
        """Load genotype data in most efficient format available
        
        Returns:
            GenotypeMatrix object optimized for performance
        """
        
        if self.data_path.suffix == '.npz':
            # Load from metadata file
            print(f"📊 Loading from optimized format: {self.data_path}")
            return load_from_metadata(self.data_path, self.precompute_alleles)
        
        elif self.data_path.suffix == '.csv':
            # Check if optimized version exists
            metadata_path = self.data_path.with_name(
                f"{self.data_path.stem}_metadata.npz"
            )
            
            if metadata_path.exists() and not self.force_conversion:
                print(f"📊 Found optimized format, loading: {metadata_path}")
                return load_from_metadata(metadata_path, self.precompute_alleles)
            else:
                print(f"📊 Converting CSV to optimized format...")
                conversion_result = convert_csv_to_optimized_format(self.data_path)
                return load_from_metadata(
                    conversion_result['metadata_path'], 
                    self.precompute_alleles
                )
        
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")


def estimate_memory_usage(shape: Tuple[int, int], 
                         dtype: np.dtype = np.int8) -> dict:
    """Estimate memory usage for genotype data
    
    Args:
        shape: (n_individuals, n_markers)
        dtype: Data type for storage
    
    Returns:
        Dictionary with memory estimates
    """
    n_individuals, n_markers = shape
    bytes_per_element = np.dtype(dtype).itemsize
    
    # Raw genotype data
    genotype_bytes = n_individuals * n_markers * bytes_per_element
    
    # Major alleles array
    major_alleles_bytes = n_markers * bytes_per_element
    
    # Working memory for batch processing (assuming 1000 marker batches)
    batch_size = min(1000, n_markers)
    batch_bytes = n_individuals * batch_size * 8  # float64 for computations
    
    total_bytes = genotype_bytes + major_alleles_bytes + batch_bytes
    
    return {
        'genotype_mb': genotype_bytes / (1024**2),
        'major_alleles_mb': major_alleles_bytes / (1024**2),
        'batch_working_mb': batch_bytes / (1024**2),
        'total_mb': total_bytes / (1024**2),
        'total_gb': total_bytes / (1024**3)
    }


def create_test_memmap(output_path: Union[str, Path],
                      n_individuals: int = 1000,
                      n_markers: int = 10000,
                      missing_rate: float = 0.05,
                      dtype: np.dtype = np.int8) -> Tuple[Path, Tuple[int, int]]:
    """Create test memory-mapped genotype file for performance testing
    
    Args:
        output_path: Output path for test file
        n_individuals: Number of individuals
        n_markers: Number of markers
        missing_rate: Proportion of missing data (-9 values)
        dtype: Data type for storage
    
    Returns:
        Tuple of (memmap_file_path, shape)
    """
    output_path = Path(output_path)
    shape = (n_individuals, n_markers)
    
    print(f"🧪 Creating test memory-mapped file: {output_path}")
    print(f"📊 Shape: {n_individuals} individuals × {n_markers} markers")
    print(f"📊 Missing data rate: {missing_rate:.1%}")
    
    # Create memory-mapped array
    memmap_array = np.memmap(output_path, dtype=dtype, mode='w+', shape=shape)
    
    # Generate realistic genotype data
    # Genotypes: 0, 1, 2 (homozygous ref, heterozygous, homozygous alt)
    genotypes = np.random.choice([0, 1, 2], size=shape, p=[0.25, 0.5, 0.25])
    
    # Add missing data
    if missing_rate > 0:
        n_missing = int(n_individuals * n_markers * missing_rate)
        missing_indices = np.random.choice(
            n_individuals * n_markers, 
            size=n_missing, 
            replace=False
        )
        genotypes.flat[missing_indices] = -9
    
    # Store to memmap
    memmap_array[:] = genotypes.astype(dtype)
    
    # Flush to disk
    del memmap_array
    
    print(f"✅ Test file created: {output_path}")
    print(f"💾 File size: {output_path.stat().st_size / (1024**2):.1f} MB")
    
    return output_path, shape
