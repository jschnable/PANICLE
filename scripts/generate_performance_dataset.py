#!/usr/bin/env python3
"""
Advanced GWAS Simulation Script with Population Structure

This script generates large-scale simulated datasets for GWAS performance testing.
It includes configurable population structure, realistic linkage patterns, and
multiple phenotypic traits with known QTN effects.

Features:
- Configurable population structure with FST-based differentiation
- Realistic linkage disequilibrium patterns within chromosomes
- Multiple phenotypic traits with shared and unique QTN effects
- Flexible MAF spectrum modeling (U-shaped, centered, empirical)
- MAF-effect size coupling with configurable architecture
- Heritability sweep functionality for benchmarking
- Study design modes for crops and inbreeding/selfing populations
- Flexible output formats compatible with PANICLE package
- Comprehensive dataset documentation and metadata

Author: Generated for PANICLE performance testing
"""

import numpy as np
import pandas as pd
import json
import argparse
import os
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional, Union
import warnings
import time
from scipy import stats
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators that do nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Global random number generator for reproducibility
# Will be properly seeded in main() function
rng_global = None

# Standalone Numba-compiled functions (must be outside class)
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _generate_chromosome_genotypes_numba_impl(freqs, positions, n_samples, seed):
        """Numba-optimized chromosome genotype generation implementation"""
        n_snps = len(freqs)
        genotypes = np.zeros((n_samples, n_snps), dtype=np.int8)

        np.random.seed(seed)

        # Generate first SNP independently for all samples
        for sample in range(n_samples):
            genotypes[sample, 0] = np.random.binomial(2, freqs[0])

        # Generate subsequent SNPs with LD
        for i in range(1, n_snps):
            distance = abs(positions[i] - positions[i-1])
            recomb_rate = min(0.5, distance / 1_000_000 * 0.01)
            ld_strength = np.exp(-recomb_rate * 100)

            for sample in range(n_samples):
                if np.random.random() < ld_strength and distance < 100_000:
                    # Strong LD - copy previous genotype with some noise
                    if np.random.random() < 0.9:
                        genotypes[sample, i] = genotypes[sample, i-1]
                    else:
                        genotypes[sample, i] = np.random.binomial(2, freqs[i])
                else:
                    # Independent generation
                    genotypes[sample, i] = np.random.binomial(2, freqs[i])

        return genotypes
else:
    def _generate_chromosome_genotypes_numba_impl(freqs, positions, n_samples, seed):
        """Fallback implementation when Numba is not available"""
        return None

class PopulationStructureSimulator:
    """Simulates population structure using Wright-Fisher model with migration"""

    def __init__(self, n_populations: int, fst: float = 0.05, rng: np.random.Generator = None):
        self.n_populations = n_populations
        self.fst = fst
        self.rng = rng or np.random.default_rng()

    def simulate_population_allele_frequencies(self, ancestral_freqs: np.ndarray) -> np.ndarray:
        """
        Simulate differentiated allele frequencies across populations using FST

        Args:
            ancestral_freqs: Array of ancestral allele frequencies

        Returns:
            Array of shape (n_populations, n_snps) with population-specific allele frequencies
        """
        n_snps = len(ancestral_freqs)

        # Vectorized calculation of population-specific frequencies using FST model
        # Balding-Nichols model: F_st = Var(p) / [p_anc * (1 - p_anc)]
        alpha = ancestral_freqs * (1 - self.fst) / self.fst + 1e-6
        beta = (1 - ancestral_freqs) * (1 - self.fst) / self.fst + 1e-6

        # Generate all populations at once using broadcasting
        pop_freqs = self.rng.beta(
            alpha[np.newaxis, :],  # Shape: (1, n_snps)
            beta[np.newaxis, :],   # Shape: (1, n_snps)
            size=(self.n_populations, n_snps)
        )

        return pop_freqs

class GWASDatasetGenerator:
    """Main class for generating comprehensive GWAS simulation datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Use new Generator interface for better random number control
        self.rng = np.random.default_rng(config['seed'])

        # Store seed chain for metadata
        self.seed_chain = {'main_seed': config['seed']}

        # Initialize population structure simulator
        if config['population_structure']['enabled']:
            pop_seed = int(self.rng.integers(0, 2**31))  # Convert to Python int
            self.seed_chain['population_seed'] = pop_seed
            self.pop_simulator = PopulationStructureSimulator(
                n_populations=config['population_structure']['n_populations'],
                fst=config['population_structure']['fst'],
                rng=np.random.default_rng(pop_seed)
            )

        # Initialize MAF spectrum
        self._setup_maf_spectrum()

        # Load empirical SFS if specified
        if (config.get('maf_spectrum') == 'empirical_sfs' and
            config.get('empirical_sfs_file')):
            self._load_empirical_sfs(config['empirical_sfs_file'])

    def _setup_maf_spectrum(self) -> None:
        """Setup MAF spectrum parameters based on configuration"""
        spectrum_type = self.config.get('maf_spectrum', 'beta_u')

        if spectrum_type == 'beta_u':
            # U-shaped distribution (default): Beta(0.5, 0.5)
            self.maf_alpha = 0.5
            self.maf_beta = 0.5
        elif spectrum_type == 'beta_centered':
            # Centered distribution (legacy): Beta(2, 2)
            self.maf_alpha = 2.0
            self.maf_beta = 2.0
        elif spectrum_type.startswith('beta_'):
            # Custom beta parameters
            self.maf_alpha = self.config.get('maf_beta_a', 0.5)
            self.maf_beta = self.config.get('maf_beta_b', 0.5)
        else:
            # Default to U-shaped if unknown
            self.maf_alpha = 0.5
            self.maf_beta = 0.5

    def _load_empirical_sfs(self, sfs_file: str) -> None:
        """Load empirical site frequency spectrum from CSV file

        Args:
            sfs_file: Path to CSV file with frequency histogram
                     Expected format: frequency,count (0-1 range)
        """
        try:
            sfs_data = pd.read_csv(sfs_file)
            if 'frequency' not in sfs_data.columns or 'count' not in sfs_data.columns:
                raise ValueError("SFS file must have 'frequency' and 'count' columns")

            frequencies = sfs_data['frequency'].values
            counts = sfs_data['count'].values

            # Normalize to probability distribution
            probabilities = counts / np.sum(counts)

            # Create interpolation function
            self.sfs_interpolator = interp1d(
                frequencies, probabilities,
                kind='linear', bounds_error=False, fill_value=0
            )

            self.sfs_frequencies = frequencies
            self.sfs_probabilities = probabilities

        except Exception as e:
            print(f"Warning: Could not load empirical SFS from {sfs_file}: {e}")
            print("Falling back to U-shaped Beta(0.5, 0.5) distribution")
            self._setup_maf_spectrum()
        
    def generate_genetic_map(self) -> pd.DataFrame:
        """Generate realistic genetic map with chromosome structure

        Creates a genetic map with realistic chromosome structure including:
        - Variable chromosome lengths (50-250 Mb)
        - Realistic SNP density with clustering patterns
        - LD block-like structure through variable spacing
        - No duplicate positions within chromosomes

        Returns:
            pd.DataFrame: Genetic map with columns ['SNP', 'Chr', 'Pos']
                - SNP: SNP identifier (e.g., 'SNP_00001')
                - Chr: Chromosome number (1 to n_chromosomes)
                - Pos: Physical position in base pairs
        """
        n_snps = self.config['n_snps']
        n_chromosomes = self.config['n_chromosomes']
        
        # Pre-allocate arrays for better performance
        snp_names = []
        chr_nums = []
        positions = []
        
        # Distribute SNPs across chromosomes (some variation in density)
        snps_per_chr = np.random.multinomial(
            n_snps, 
            np.ones(n_chromosomes) / n_chromosomes
        )
        
        snp_idx = 0
        
        for chr_num in range(1, n_chromosomes + 1):
            n_snps_chr = snps_per_chr[chr_num - 1]
            
            # Generate positions with realistic density (more SNPs in gene-rich regions)
            if n_snps_chr > 0:
                # Chromosome length varies realistically (50-250 Mb)
                chr_length = self.rng.integers(50_000_000, 250_000_000)
                
                # Generate positions with realistic spacing and no duplicates
                chr_positions = []
                
                # Calculate target spacing to distribute SNPs across chromosome
                target_spacing = chr_length // (n_snps_chr + 1)  # +1 to avoid boundary issues
                min_spacing = max(100, target_spacing // 10)  # Minimum 100bp spacing
                max_spacing = target_spacing * 3  # Allow some clustering
                
                # Start position (not too close to chromosome start)
                start_max = max(100_001, min(1_000_000, target_spacing))  # Ensure high > low
                current_pos = self.rng.integers(100_000, start_max)
                
                for i in range(n_snps_chr):
                    chr_positions.append(current_pos)
                    
                    # Variable spacing with realistic bounds
                    if self.rng.random() < 0.3:  # 30% chance of tight clustering (LD blocks)
                        spacing = self.rng.integers(min_spacing, min_spacing * 10)
                    else:  # 70% chance of regular spacing
                        spacing = self.rng.integers(min_spacing * 5, max_spacing)
                    
                    current_pos += spacing
                    
                    # If we're getting close to chromosome end, reduce spacing
                    remaining_snps = n_snps_chr - i - 1
                    if remaining_snps > 0:
                        remaining_space = chr_length - current_pos - 100_000  # Leave 100kb buffer
                        if remaining_space > 0:
                            max_allowed_spacing = remaining_space // remaining_snps
                            if spacing > max_allowed_spacing:
                                current_pos = chr_positions[-1] + min(spacing, max_allowed_spacing)
                
                # Ensure all positions are within chromosome bounds and unique
                chr_positions = [min(pos, chr_length - 1000) for pos in chr_positions]
                chr_positions = sorted(set(chr_positions))  # Remove duplicates and sort
                
                # If we lost SNPs due to duplicate removal, regenerate missing ones
                while len(chr_positions) < n_snps_chr:
                    # Find the largest gap and add a SNP there
                    gaps = []
                    for i in range(len(chr_positions) - 1):
                        gap_size = chr_positions[i+1] - chr_positions[i]
                        if gap_size > min_spacing * 2:
                            gaps.append((gap_size, i))
                    
                    if gaps:
                        # Add SNP in largest gap
                        gaps.sort(reverse=True)
                        gap_size, gap_idx = gaps[0]
                        new_pos = chr_positions[gap_idx] + gap_size // 2
                        chr_positions.insert(gap_idx + 1, new_pos)
                    else:
                        # If no suitable gaps, add at a random position with minimum spacing
                        attempts = 0
                        while attempts < 100:
                            new_pos = self.rng.integers(100_000, chr_length - 1000)
                            if all(abs(new_pos - pos) >= min_spacing for pos in chr_positions):
                                chr_positions.append(new_pos)
                                chr_positions.sort()
                                break
                            attempts += 1
                        if attempts >= 100:
                            break  # Give up if we can't find a good position
                
                # Final sort and trim to exact number needed
                chr_positions = sorted(chr_positions)[:n_snps_chr]
                
                # Batch append to lists
                snp_names.extend([f'SNP_{snp_idx + i:05d}' for i in range(n_snps_chr)])
                chr_nums.extend([chr_num] * n_snps_chr)
                positions.extend(chr_positions)
                snp_idx += n_snps_chr
        
        # Create DataFrame in one operation
        return pd.DataFrame({
            'SNP': snp_names,
            'Chr': chr_nums,
            'Pos': positions
        })

    def _generate_ancestral_frequencies(self, n_snps: int) -> np.ndarray:
        """Generate ancestral allele frequencies according to specified spectrum

        Args:
            n_snps: Number of SNPs to generate frequencies for

        Returns:
            Array of ancestral allele frequencies
        """
        spectrum_type = self.config.get('maf_spectrum', 'beta_u')

        if spectrum_type == 'empirical_sfs' and hasattr(self, 'sfs_interpolator'):
            # Sample from empirical SFS using inverse transform sampling
            uniform_samples = self.rng.random(n_snps)
            cumulative_probs = np.cumsum(self.sfs_probabilities)
            cumulative_probs /= cumulative_probs[-1]  # Normalize

            # Inverse transform sampling
            indices = np.searchsorted(cumulative_probs, uniform_samples)
            indices = np.clip(indices, 0, len(self.sfs_frequencies) - 1)
            ancestral_freqs = self.sfs_frequencies[indices]

            # Add small amount of jitter to avoid exact duplicates
            jitter = self.rng.normal(0, 0.001, n_snps)
            ancestral_freqs = np.clip(ancestral_freqs + jitter, 0.001, 0.999)

        else:
            # Beta distribution sampling
            ancestral_freqs = self.rng.beta(self.maf_alpha, self.maf_beta, n_snps)
            # Ensure no extreme values that could cause numerical issues
            ancestral_freqs = np.clip(ancestral_freqs, 0.001, 0.999)

        return ancestral_freqs

    def generate_genotype_matrix(self, genetic_map: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Generate genotype matrix with population structure and linkage disequilibrium

        Creates a realistic genotype matrix with:
        - Population structure (if enabled) using Wright-Fisher model
        - Realistic linkage disequilibrium patterns within chromosomes
        - Configurable MAF spectrum (U-shaped, centered, or empirical)
        - Study design effects (inbreeding for crop populations)
        - Controlled missing data patterns

        Args:
            genetic_map: DataFrame with SNP positions and chromosome assignments

        Returns:
            Tuple containing:
                - np.ndarray: Genotype matrix (samples × SNPs) with values {0, 1, 2, -9}
                  where -9 indicates missing data
                - List[str]: Sample identifiers
        """
        n_samples = self.config['n_samples']
        n_snps = len(genetic_map)
        
        # Generate sample IDs
        sample_ids = [f'ID{i:04d}' for i in range(n_samples)]
        
        # Pre-allocate genotype matrix
        genotypes = np.zeros((n_samples, n_snps), dtype=np.int8)
        
        # Pre-process chromosome information for vectorized operations
        chr_info = self._preprocess_chromosome_info(genetic_map)

        # Generate ancestral allele frequencies according to spectrum
        ancestral_freqs = self._generate_ancestral_frequencies(n_snps)

        if self.config['population_structure']['enabled']:
            # Simulate population structure
            pop_config = self.config['population_structure']
            pop_sizes = pop_config['population_sizes']
            
            # Get population-specific allele frequencies from ancestral
            pop_freqs = self.pop_simulator.simulate_population_allele_frequencies(ancestral_freqs)
            
            sample_idx = 0
            for pop_idx, pop_size in enumerate(pop_sizes):
                if sample_idx >= n_samples:
                    break
                    
                end_idx = min(sample_idx + pop_size, n_samples)
                pop_samples = end_idx - sample_idx
                
                # Generate all samples for this population at once
                pop_genotypes = self._generate_population_genotypes_vectorized(
                    pop_freqs[pop_idx, :], chr_info, pop_samples
                )
                genotypes[sample_idx:end_idx, :] = pop_genotypes
                sample_idx = end_idx
        else:
            # No population structure - use ancestral allele frequencies directly
            genotypes = self._generate_population_genotypes_vectorized(
                ancestral_freqs, chr_info, n_samples
            )
        
        # Apply study design modifications (inbreeding/selfing)
        if self.config.get('study_design') == 'inbred':
            genotypes = self._apply_inbreeding(genotypes)

        # Vectorized missing data assignment
        if self.config['missing_data_rate'] > 0:
            n_missing = int(n_samples * n_snps * self.config['missing_data_rate'])
            missing_mask = self.rng.random((n_samples, n_snps)) < (n_missing / (n_samples * n_snps))
            genotypes[missing_mask] = -9

        # Store ancestral frequencies for effect size coupling
        self.ancestral_freqs = ancestral_freqs

        return genotypes, sample_ids

    def _apply_inbreeding(self, genotypes: np.ndarray) -> np.ndarray:
        """Apply inbreeding effects to genotype matrix

        Args:
            genotypes: Original genotype matrix

        Returns:
            Modified genotype matrix with inbreeding effects
        """
        inbreeding_f = self.config.get('inbreeding_f', 0.95)
        n_samples, n_snps = genotypes.shape

        # For each sample, collapse to near-homozygous genotypes
        inbred_genotypes = genotypes.copy()

        for sample_idx in range(n_samples):
            for snp_idx in range(n_snps):
                if genotypes[sample_idx, snp_idx] == -9:  # Skip missing data
                    continue

                # With probability F, make homozygous
                if self.rng.random() < inbreeding_f:
                    current_geno = genotypes[sample_idx, snp_idx]
                    if current_geno == 1:  # Heterozygote
                        # Randomly choose 0 or 2 (homozygous)
                        inbred_genotypes[sample_idx, snp_idx] = self.rng.choice([0, 2])
                    # Homozygotes (0, 2) remain unchanged

        return inbred_genotypes

    def _preprocess_chromosome_info(self, genetic_map: pd.DataFrame) -> Dict[str, Any]:
        """Pre-process chromosome information for efficient genotype generation

        Organizes genetic map data by chromosome for vectorized operations.

        Args:
            genetic_map: DataFrame with SNP map information

        Returns:
            Dict mapping chromosome numbers to:
                - indices: Array indices for SNPs on this chromosome
                - positions: Physical positions of SNPs
                - n_snps: Number of SNPs on chromosome
        """
        chr_info = {}
        for chr_num in genetic_map['Chr'].unique():
            chr_mask = genetic_map['Chr'] == chr_num
            chr_indices = np.where(chr_mask)[0]
            chr_positions = genetic_map.loc[chr_mask, 'Pos'].values
            
            if len(chr_indices) > 0:
                chr_info[chr_num] = {
                    'indices': chr_indices,
                    'positions': chr_positions,
                    'n_snps': len(chr_indices)
                }
        return chr_info
    
    def _generate_population_genotypes_vectorized(self, allele_freqs: np.ndarray,
                                                 chr_info: Dict[str, Any],
                                                 n_samples: int) -> np.ndarray:
        """Generate genotypes for multiple samples using vectorized operations

        Efficiently generates genotypes across all chromosomes with realistic
        linkage disequilibrium patterns.

        Args:
            allele_freqs: Allele frequencies for all SNPs
            chr_info: Pre-processed chromosome information
            n_samples: Number of samples to generate

        Returns:
            np.ndarray: Genotype matrix (samples × SNPs)
        """
        n_snps = len(allele_freqs)
        genotypes = np.zeros((n_samples, n_snps), dtype=np.int8)
        
        for chr_num, info in chr_info.items():
            chr_indices = info['indices']
            chr_positions = info['positions']
            chr_freqs = allele_freqs[chr_indices]
            
            if len(chr_indices) == 0:
                continue
                
            # Generate all samples for this chromosome at once
            chr_genotypes = self._generate_chromosome_genotypes_vectorized(
                chr_freqs, chr_positions, n_samples
            )
            genotypes[:, chr_indices] = chr_genotypes
        
        return genotypes
    
    def _generate_chromosome_genotypes_vectorized(self, freqs: np.ndarray,
                                                 positions: np.ndarray,
                                                 n_samples: int) -> np.ndarray:
        """Generate genotypes for a single chromosome with realistic LD

        Uses distance-based linkage disequilibrium model:
        - LD strength = exp(-recombination_rate × 100)
        - Strong LD for SNPs < 100kb apart
        - Recombination rate = distance / 100 cM per Mb

        Args:
            freqs: Allele frequencies for SNPs on this chromosome
            positions: Physical positions of SNPs
            n_samples: Number of samples to generate

        Returns:
            np.ndarray: Chromosome genotypes (samples × SNPs_on_chr)
        """
        n_snps = len(freqs)
        if n_snps == 0:
            return np.array([], dtype=np.int8).reshape(n_samples, 0)
        
        if NUMBA_AVAILABLE:
            return self._generate_chromosome_genotypes_numba(
                freqs, positions, n_samples, self.rng.integers(0, 2**31)
            )
        else:
            return self._generate_chromosome_genotypes_numpy(
                freqs, positions, n_samples
            )
    
    def _generate_chromosome_genotypes_numpy(self, freqs: np.ndarray,
                                           positions: np.ndarray,
                                           n_samples: int) -> np.ndarray:
        """NumPy-based chromosome genotype generation with LD

        Fallback implementation when Numba is not available.
        Implements the same LD model as the Numba version but with
        pure NumPy operations.

        Args:
            freqs: Allele frequencies for chromosome SNPs
            positions: Physical positions in base pairs
            n_samples: Number of samples to generate

        Returns:
            np.ndarray: Genotype matrix for chromosome
        """
        n_snps = len(freqs)
        genotypes = np.zeros((n_samples, n_snps), dtype=np.int8)
        
        # Generate first SNP independently for all samples
        genotypes[:, 0] = self.rng.binomial(2, freqs[0], n_samples)
        
        # Pre-calculate LD parameters
        if n_snps > 1:
            distances = np.abs(np.diff(positions))
            recomb_rates = np.minimum(0.5, distances / 1_000_000 * 0.01)
            ld_strengths = np.exp(-recomb_rates * 100)
            
            # Vectorized LD generation
            for i in range(1, n_snps):
                distance = distances[i-1]
                ld_strength = ld_strengths[i-1]
                
                if distance < 100_000:
                    # Use LD with previous SNP
                    copy_mask = self.rng.random(n_samples) < (ld_strength * 0.9)
                    genotypes[copy_mask, i] = genotypes[copy_mask, i-1]

                    # Independent generation for remaining samples
                    independent_mask = ~copy_mask
                    if np.sum(independent_mask) > 0:
                        genotypes[independent_mask, i] = self.rng.binomial(
                            2, freqs[i], np.sum(independent_mask)
                        )
                else:
                    # Independent generation for all samples
                    genotypes[:, i] = self.rng.binomial(2, freqs[i], n_samples)
        
        return genotypes
    
    def _generate_chromosome_genotypes_numba(self, freqs: np.ndarray,
                                           positions: np.ndarray,
                                           n_samples: int,
                                           seed: int) -> np.ndarray:
        """Numba-optimized chromosome genotype generation"""
        if NUMBA_AVAILABLE:
            return _generate_chromosome_genotypes_numba_impl(freqs, positions, n_samples, seed)
        else:
            return self._generate_chromosome_genotypes_numpy(freqs, positions, n_samples)
    
    def select_qtns(self, genetic_map: pd.DataFrame) -> Tuple[List[int], np.ndarray]:
        """Select QTN positions ensuring even chromosome distribution with MAF-effect coupling"""
        n_qtns = self.config['n_qtns']
        n_chromosomes = self.config['n_chromosomes']

        # Distribute QTNs across chromosomes
        qtns_per_chr = np.random.multinomial(
            n_qtns,
            np.ones(n_chromosomes) / n_chromosomes
        )

        qtn_indices = []

        for chr_num in range(1, n_chromosomes + 1):
            chr_mask = genetic_map['Chr'] == chr_num
            chr_indices = genetic_map.index[chr_mask].tolist()

            n_qtns_chr = qtns_per_chr[chr_num - 1]
            if n_qtns_chr > 0 and len(chr_indices) > 0:
                # Select QTNs from this chromosome
                selected = self.rng.choice(
                    chr_indices,
                    min(n_qtns_chr, len(chr_indices)),
                    replace=False
                )
                qtn_indices.extend(selected)

        # Apply MAF-effect coupling if configured
        kappa = self.config.get('effect_maf_coupling_kappa', 0.0)
        if kappa > 0 and hasattr(self, 'ancestral_freqs') and len(qtn_indices) > 0:
            qtn_mafs = self.ancestral_freqs[qtn_indices]
            # Apply coupling: |β| ∝ [MAF * (1-MAF)]^(-κ/2)
            # This makes rare alleles have larger effect sizes
            maf_variance = qtn_mafs * (1 - qtn_mafs)
            maf_variance = np.maximum(maf_variance, 0.001)  # Prevent division by zero

            # Calculate coupling factor BEFORE generating effects
            coupling_factor = np.power(maf_variance, -kappa / 2)

            # Normalize coupling factor to have mean=1 and reasonable range
            coupling_factor = coupling_factor / np.mean(coupling_factor)
            # Clip to reasonable range (prevents extreme outliers while maintaining correlation)
            coupling_factor = np.clip(coupling_factor, 0.1, 10.0)

            # Generate effects with variance proportional to coupling factor
            # This ensures the MAF-effect relationship is built into the generation
            qtn_effects = self.rng.normal(0, 0.3 * coupling_factor, len(qtn_indices))

            # When coupling is enabled, do NOT add large-effect QTNs
            # as they would break the MAF-effect correlation
        else:
            # No coupling - generate standard effects
            qtn_effects = self.rng.normal(0, 0.3, len(qtn_indices))

            # Add traditional large-effect QTNs
            n_large_effect = max(1, len(qtn_indices) // 20)
            large_effect_indices = self.rng.choice(len(qtn_indices), n_large_effect, replace=False)
            for idx in large_effect_indices:
                qtn_effects[idx] = self.rng.normal(0, 1.0)  # Larger effect

        return qtn_indices, qtn_effects
    
    def generate_phenotypes(self, genotypes: np.ndarray, qtn_indices: List[int], 
                          qtn_effects: np.ndarray, sample_ids: List[str]) -> pd.DataFrame:
        """Generate multiple correlated phenotypes with QTN effects using vectorized operations"""
        n_samples = len(sample_ids)
        n_traits = self.config['n_traits']
        n_qtns = len(qtn_indices)
        
        # Create trait correlation structure
        if n_traits > 1:
            correlation_matrix = self._generate_trait_correlation_matrix(n_traits)
        else:
            correlation_matrix = np.array([[1.0]])
        
        # Initialize phenotype matrix
        phenotypes = np.zeros((n_samples, n_traits))
        
        if n_qtns > 0:
            # Extract QTN genotypes once (vectorized)
            qtn_genotypes = genotypes[:, qtn_indices]  # Shape: (n_samples, n_qtns)
            
            # Create QTN effect matrix for all traits
            trait_qtn_effects = np.zeros((n_qtns, n_traits))
            
            # Generate trait-specific effects vectorized
            for trait_idx in range(n_traits):
                # Random selection of which QTNs affect this trait
                qtn_affects_trait = self.rng.random(n_qtns) < 0.7
                
                # Generate trait-specific effect modifications
                trait_modifications = self.rng.normal(1.0, 0.3, n_qtns)
                
                # Apply effects only where QTN affects trait
                trait_qtn_effects[:, trait_idx] = (
                    qtn_effects * trait_modifications * qtn_affects_trait
                )
            
            # Vectorized computation of genetic values
            # Handle missing values by masking
            qtn_genotypes_clean = np.where(qtn_genotypes == -9, 0, qtn_genotypes)
            missing_mask = qtn_genotypes == -9
            
            # Matrix multiplication: (n_samples, n_qtns) @ (n_qtns, n_traits) = (n_samples, n_traits)
            genetic_values = qtn_genotypes_clean @ trait_qtn_effects
            
            # Zero out contributions from missing genotypes
            for trait_idx in range(n_traits):
                for qtn_idx in range(n_qtns):
                    genetic_values[missing_mask[:, qtn_idx], trait_idx] = 0
            
            phenotypes = genetic_values
        
        # Add correlated environmental effects
        environmental_noise = self.rng.multivariate_normal(
            np.zeros(n_traits),
            correlation_matrix * 0.8,
            size=n_samples
        )
        
        # Add population structure effects if enabled
        if self.config['population_structure']['enabled']:
            pop_effects = self._generate_population_effects(n_traits, sample_ids)
            phenotypes += pop_effects
        
        # Combine genetic and environmental components
        phenotypes += environmental_noise
        
        # Vectorized standardization
        phenotypes = (phenotypes - np.mean(phenotypes, axis=0)) / np.std(phenotypes, axis=0)
        
        # Create DataFrame
        trait_names = [f'trait{i+1}' for i in range(n_traits)]
        phenotype_df = pd.DataFrame(phenotypes, columns=trait_names)
        phenotype_df.insert(0, 'Taxa', sample_ids)
        
        return phenotype_df
    
    def _generate_trait_correlation_matrix(self, n_traits: int) -> np.ndarray:
        """Generate realistic correlation structure between traits"""
        # Start with identity matrix
        correlation_matrix = np.eye(n_traits)
        
        # Add some correlations between traits
        for i in range(n_traits):
            for j in range(i + 1, n_traits):
                # Random correlation between -0.5 and 0.8
                corr = self.rng.uniform(-0.5, 0.8)
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.1)  # Minimum eigenvalue
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Normalize to correlation matrix
        d_inv = np.diag(1 / np.sqrt(np.diag(correlation_matrix)))
        correlation_matrix = d_inv @ correlation_matrix @ d_inv
        
        return correlation_matrix
    
    def _generate_population_effects(self, n_traits: int, sample_ids: List[str]) -> np.ndarray:
        """Generate population-specific phenotype effects

        Creates population structure in phenotypes by assigning different
        population means for each trait. This simulates population stratification
        that can confound GWAS if not properly controlled.

        Args:
            n_traits: Number of phenotypic traits
            sample_ids: List of sample identifiers for assignment

        Returns:
            np.ndarray: Population effects matrix (samples × traits)
        """
        pop_config = self.config['population_structure']
        pop_sizes = pop_config['population_sizes']
        n_populations = len(pop_sizes)
        
        # Generate population means for each trait
        pop_means = self.rng.normal(0, 0.5, (n_populations, n_traits))
        
        # Assign samples to populations and apply effects
        pop_effects = np.zeros((len(sample_ids), n_traits))
        sample_idx = 0
        
        for pop_idx, pop_size in enumerate(pop_sizes):
            for _ in range(pop_size):
                if sample_idx >= len(sample_ids):
                    break
                pop_effects[sample_idx, :] = pop_means[pop_idx, :]
                sample_idx += 1
        
        return pop_effects
    
    def create_dataset_summary(self, genetic_map: pd.DataFrame,
                             qtn_indices: List[int], qtn_effects: np.ndarray,
                             phenotype_df: pd.DataFrame, genotypes: np.ndarray,
                             output_dir: Path) -> Dict[str, Any]:
        """Create comprehensive dataset summary with metadata and design summary

        Args:
            genetic_map: SNP map DataFrame
            qtn_indices: Indices of causal variants
            qtn_effects: Effect sizes for QTNs
            phenotype_df: Phenotype data
            genotypes: Genotype matrix (for calculating design summary)
            output_dir: Output directory path

        Returns:
            Dict with comprehensive dataset metadata
        """

        # Calculate population structure info
        if self.config['population_structure']['enabled']:
            pop_structure = {
                "n_populations": self.config['population_structure']['n_populations'],
                "population_sizes": self.config['population_structure']['population_sizes'],
                "fst": self.config['population_structure']['fst']
            }
        else:
            pop_structure = {
                "n_populations": 1,
                "population_sizes": [self.config['n_samples']],
                "fst": 0.0
            }

        # Calculate trait summaries
        trait_summaries = {}
        for col in phenotype_df.columns:
            if col != 'Taxa':
                trait_data = phenotype_df[col]
                trait_summaries[col] = {
                    "mean": float(np.mean(trait_data)),
                    "sd": float(np.std(trait_data)),
                    "min": float(np.min(trait_data)),
                    "max": float(np.max(trait_data))
                }

        # Calculate design summary metrics
        design_summary_metrics = self._calculate_design_summary(genotypes, qtn_indices)

        summary = {
            "dataset_params": self.config,
            "dimensions": {
                "n_samples": self.config['n_samples'],
                "n_snps": self.config['n_snps'],
                "n_qtns": len(qtn_indices),
                "n_traits": self.config['n_traits'],
                "n_chromosomes": self.config['n_chromosomes']
            },
            "qtns": {
                "indices": [int(x) for x in qtn_indices],  # Convert to Python int
                "effects": qtn_effects.tolist(),
                "effect_range": [float(np.min(qtn_effects)), float(np.max(qtn_effects))]
            },
            "population_structure": pop_structure,
            "trait_summaries": trait_summaries,
            "design_summary": design_summary_metrics,
            "missing_data_rate": self.config['missing_data_rate'],
            "seed_chain": self.seed_chain,
            "target_heritability": self.config.get('heritability', 0.7),
            "generation_timestamp": pd.Timestamp.now().isoformat(),
            "output_directory": str(output_dir)
        }

        return summary

    def _calculate_design_summary(self, genotypes: np.ndarray, qtn_indices: List[int]) -> Dict[str, Any]:
        """Calculate study design summary metrics

        Args:
            genotypes: Genotype matrix
            qtn_indices: Indices of causal variants

        Returns:
            Dict with design metrics (heterozygosity, MAF, etc.)
        """
        n_samples, n_snps = genotypes.shape

        # Calculate observed heterozygosity
        valid_genotypes = genotypes[genotypes != -9]
        het_count = np.sum(valid_genotypes == 1)
        total_valid = len(valid_genotypes)
        observed_het = het_count / total_valid if total_valid > 0 else 0.0

        # Calculate expected heterozygosity under HWE
        # For each SNP, get allele freq and expected het = 2*p*(1-p)
        expected_het_values = []
        for snp_idx in range(n_snps):
            snp_data = genotypes[:, snp_idx]
            valid = snp_data[snp_data != -9]
            if len(valid) > 0:
                allele_freq = np.mean(valid) / 2.0
                expected_het_values.append(2 * allele_freq * (1 - allele_freq))

        expected_het = np.mean(expected_het_values) if expected_het_values else 0.0

        # Calculate inbreeding coefficient F = 1 - (H_obs / H_exp)
        f_stat = 1.0 - (observed_het / expected_het) if expected_het > 0 else 0.0

        return {
            "observed_heterozygosity": float(observed_het),
            "expected_heterozygosity": float(expected_het),
            "inbreeding_coefficient": float(f_stat),
            "study_design": self.config.get('study_design', 'unknown')
        }
    
    def _save_genotype_csv_optimized(self, genotypes: np.ndarray, sample_ids: List[str], output_dir: Path) -> None:
        """Optimized CSV writing without DataFrame overhead

        Writes genotype matrix directly to CSV using Python's csv module
        to avoid pandas DataFrame memory overhead for large datasets.

        Args:
            genotypes: Genotype matrix (samples × SNPs)
            sample_ids: Sample identifiers for row names
            output_dir: Output directory path
        """
        import csv
        
        csv_path = output_dir / 'genotype_numeric.csv'
        n_samples, n_snps = genotypes.shape
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = [''] + [f'SNP_{i:05d}' for i in range(n_snps)]
            writer.writerow(header)
            
            # Write data rows
            for i, sample_id in enumerate(sample_ids):
                row = [sample_id] + ['' if x == -9 else str(x) for x in genotypes[i, :]]
                writer.writerow(row)
    
    def save_dataset(self, output_dir: Path, genetic_map: pd.DataFrame,
                    genotypes: np.ndarray, sample_ids: List[str],
                    qtn_indices: List[int], qtn_effects: np.ndarray,
                    phenotype_df: pd.DataFrame) -> None:
        """Save complete dataset in multiple formats with comprehensive metadata

        Saves dataset in formats compatible with PANICLE and other GWAS tools:
        - Compressed NumPy arrays for fast loading
        - CSV files for compatibility
        - Parquet files for efficient storage
        - JSON metadata with validation statistics
        - Truth tables for benchmarking
        - Null phenotypes for testing

        Args:
            output_dir: Directory to save files
            genetic_map: SNP map information
            genotypes: Genotype matrix
            sample_ids: Sample identifiers
            qtn_indices: Indices of causal variants
            qtn_effects: Effect sizes for QTNs
            phenotype_df: Phenotype data
        """
        import time
        
        print(f"Saving dataset to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save genetic map (optimized)
        map_start = time.time()
        genetic_map.to_csv(output_dir / 'map.csv', index=False)
        map_time = time.time() - map_start
        print(f"  ✓ Saved genetic map: {len(genetic_map)} SNPs in {map_time:.1f}s")
        
        # 2. Save genotype matrix in optimized format
        # Save as compressed numpy array for faster loading
        np.savez_compressed(output_dir / 'genotypes.npz', 
                           genotypes=genotypes,
                           sample_ids=sample_ids,
                           snp_ids=[f'SNP_{i:05d}' for i in range(genotypes.shape[1])])
        
        # Optimized file I/O with timing
        io_start = time.time()
        
        # Fast Parquet save (no DataFrame creation needed for large datasets)
        try:
            print(f"  ✓ Saving Parquet format (fast binary)...")
            # Create minimal DataFrame for Parquet (much faster than full DataFrame)
            if genotypes.shape[0] * genotypes.shape[1] > 10_000_000:  # Large dataset
                # For very large datasets, save directly as NPZ and skip DataFrame creation
                print(f"    Large dataset detected - skipping DataFrame creation for speed")
            else:
                # For smaller datasets, create DataFrame
                genotype_df = pd.DataFrame(genotypes, index=sample_ids, 
                                         columns=[f'SNP_{i:05d}' for i in range(genotypes.shape[1])])
                genotype_df.to_parquet(output_dir / 'genotype_numeric.parquet', compression='snappy')
                
            parquet_time = time.time() - io_start
            print(f"  ✓ Saved genotypes (Parquet): {genotypes.shape[0]} × {genotypes.shape[1]} in {parquet_time:.1f}s")
            
        except (ImportError, Exception) as e:
            print(f"  ⚠ Parquet save failed: {e}")
        
        # CSV save (optional for compatibility - can be skipped for speed)
        csv_start = time.time()
        if genotypes.shape[0] * genotypes.shape[1] <= 200_000_000:  # Allow CSV for medium datasets
            print(f"  ✓ Saving CSV format (compatibility)...")
            # Use optimized CSV writing
            self._save_genotype_csv_optimized(genotypes, sample_ids, output_dir)
            csv_time = time.time() - csv_start
            print(f"  ✓ Saved genotypes (CSV): {genotypes.shape[0]} × {genotypes.shape[1]} in {csv_time:.1f}s")
        else:
            print(f"  ⚠ Skipping CSV save for very large dataset (>200M genotypes) - use NPZ for loading")
        
        # 3. Save phenotype data
        phenotype_df.to_csv(output_dir / 'phenotype.csv', index=False)
        print(f"  ✓ Saved phenotypes: {len(phenotype_df)} samples × {len(phenotype_df.columns)-1} traits")
        
        # 4. Save sample names (fast write)
        sample_time = time.time()
        with open(output_dir / 'sample_names.csv', 'w') as f:
            f.write('Sample_ID\n')
            f.write('\n'.join(sample_ids))
        print(f"  ✓ Saved sample names: {len(sample_ids)} samples in {time.time() - sample_time:.1f}s")
        
        # 5. Save SNP names (fast generation and write)
        snp_time = time.time()
        with open(output_dir / 'snp_names.csv', 'w') as f:
            f.write('SNP_ID\n')
            for i in range(len(genetic_map)):
                f.write(f'SNP_{i:05d}\n')
        print(f"  ✓ Saved SNP names: {len(genetic_map)} SNPs in {time.time() - snp_time:.1f}s")
        
        # 6. Save true QTN information (optimized)
        qtn_time = time.time()
        with open(output_dir / 'true_qtns.csv', 'w') as f:
            f.write('QTN_Index,SNP_Name,Chr,Pos,True_Effect\n')
            for i, qtn_idx in enumerate(qtn_indices):
                qtn_info = genetic_map.iloc[qtn_idx]
                f.write(f'{qtn_idx},{qtn_info["SNP"]},{qtn_info["Chr"]},{qtn_info["Pos"]},{qtn_effects[i]}\n')
        print(f"  ✓ Saved QTN info: {len(qtn_indices)} true QTNs in {time.time() - qtn_time:.1f}s")
        
        # 7. Save dataset summary
        summary_time = time.time()
        summary = self.create_dataset_summary(
            genetic_map, qtn_indices, qtn_effects, phenotype_df, genotypes, output_dir
        )

        with open(output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved dataset summary in {time.time() - summary_time:.1f}s")

        # 8. Save design summary as TSV
        design_time = time.time()
        design_summary = summary.get('design_summary', {})
        with open(output_dir / 'design_summary.tsv', 'w') as f:
            f.write('Metric\tValue\n')
            for key, value in design_summary.items():
                f.write(f'{key}\t{value}\n')
        print(f"  ✓ Saved design summary in {time.time() - design_time:.1f}s")

        # 9. Save truth table (QTN info in different format)
        truth_time = time.time()
        with open(output_dir / 'truth.tsv', 'w') as f:
            f.write('SNP\tChr\tPos\tEffect\n')
            for i, qtn_idx in enumerate(qtn_indices):
                qtn_info = genetic_map.iloc[qtn_idx]
                f.write(f'{qtn_info["SNP"]}\t{qtn_info["Chr"]}\t{qtn_info["Pos"]}\t{qtn_effects[i]}\n')
        print(f"  ✓ Saved truth table in {time.time() - truth_time:.1f}s")

        # 10. Save null phenotype (all zeros for negative control)
        null_time = time.time()
        null_phenotype_df = phenotype_df.copy()
        for col in null_phenotype_df.columns:
            if col != 'Taxa':
                null_phenotype_df[col] = 0.0
        null_phenotype_df.to_csv(output_dir / 'phenotype_null.csv', index=False)
        print(f"  ✓ Saved null phenotype in {time.time() - null_time:.1f}s")

        print(f"\nDataset generation complete!")
        print(f"Output directory: {output_dir}")
        print(f"Total files created: 10")

def create_config(args) -> Dict[str, Any]:
    """Create configuration dictionary from command line arguments"""
    
    # Parse population sizes if provided
    pop_sizes = [int(x) for x in args.population_sizes.split(',')]
    
    config = {
        'n_samples': args.n_samples,
        'n_snps': args.n_snps, 
        'n_qtns': args.n_qtns,
        'n_chromosomes': args.n_chromosomes,
        'n_traits': args.n_traits,
        'seed': args.seed,
        'missing_data_rate': args.missing_data_rate,
        'population_structure': {
            'enabled': args.population_structure,
            'n_populations': len(pop_sizes),
            'population_sizes': pop_sizes,
            'fst': args.fst
        },
        'output_dir': args.output_dir
    }
    
    return config

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(
        description='Generate large-scale GWAS simulation dataset with population structure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core dataset parameters
    parser.add_argument('--n-samples', type=int, default=2000,
                       help='Number of individuals/samples')
    parser.add_argument('--n-snps', type=int, default=200000,
                       help='Number of genetic markers (SNPs)')
    parser.add_argument('--n-qtns', type=int, default=100,
                       help='Number of true QTNs (causal variants)')
    parser.add_argument('--n-chromosomes', type=int, default=10,
                       help='Number of chromosomes')
    parser.add_argument('--n-traits', type=int, default=10,
                       help='Number of phenotypic traits')
    
    # Population structure parameters
    parser.add_argument('--population-structure', action='store_true', default=True,
                       help='Enable population structure simulation')
    parser.add_argument('--population-sizes', type=str, default='600,800,600',
                       help='Comma-separated population sizes (e.g., "600,800,600")')
    parser.add_argument('--fst', type=float, default=0.05,
                       help='Population differentiation (FST)')
    
    # MAF spectrum and genetic architecture
    parser.add_argument('--maf-spectrum', choices=['beta_u', 'beta_centered', 'empirical_sfs'],
                       default='beta_u',
                       help='MAF spectrum model: beta_u (U-shaped, default), beta_centered (legacy), empirical_sfs')
    parser.add_argument('--maf-beta-a', type=float, default=0.5,
                       help='Beta distribution alpha parameter for MAF spectrum')
    parser.add_argument('--maf-beta-b', type=float, default=0.5,
                       help='Beta distribution beta parameter for MAF spectrum')
    parser.add_argument('--empirical-sfs-file', type=str,
                       help='Path to empirical SFS CSV file (frequency,count columns)')
    parser.add_argument('--effect-maf-coupling-kappa', type=float, default=0.25,
                       help='MAF-effect coupling parameter: |β| ∝ MAF^(-κ)')
    parser.add_argument('--effect-size-dist', choices=['normal', 't'], default='normal',
                       help='Distribution for base effect sizes')
    parser.add_argument('--effect-size-scale', type=float, default=0.3,
                       help='Scale parameter for effect size distribution')

    # Heritability control
    parser.add_argument('--heritability', type=float, default=0.7,
                       help='Target heritability (proportion of variance due to genetics)')

    # Study design parameters
    parser.add_argument('--study-design', choices=['outbred', 'inbred'], default='inbred',
                       help='Study design: outbred (human-like) or inbred (crop-like)')
    parser.add_argument('--inbreeding-f', type=float, default=0.95,
                       help='Inbreeding coefficient for inbred design')

    # Data quality parameters
    parser.add_argument('--missing-data-rate', type=float, default=0.02,
                       help='Proportion of missing genotype data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='performance_test_huge',
                       help='Output directory name')

    # Convenience scenarios
    parser.add_argument('--scenario', choices=['hard_mode'],
                       help='Predefined challenging scenarios for benchmarking')

    args = parser.parse_args()

    # Apply scenario presets
    if args.scenario == 'hard_mode':
        args.maf_spectrum = 'beta_u'
        args.effect_maf_coupling_kappa = 0.5
        args.heritability = 0.2
        args.study_design = 'inbred'
        args.inbreeding_f = 0.98
        args.n_qtns = 200
        print("Applied 'hard_mode' scenario preset")

    # Set global random seed
    rng_global = np.random.default_rng(args.seed)
    
    # Validate arguments
    if sum([int(x) for x in args.population_sizes.split(',')]) != args.n_samples:
        print(f"Warning: Population sizes sum to {sum([int(x) for x in args.population_sizes.split(',')])} "
              f"but n_samples is {args.n_samples}. Adjusting n_samples.")
        args.n_samples = sum([int(x) for x in args.population_sizes.split(',')])
    
    # Create configuration
    config = create_config(args)
    
    # Print configuration
    print("=== GWAS Dataset Generation Configuration ===")
    print(f"Samples: {config['n_samples']:,}")
    print(f"SNPs: {config['n_snps']:,}")
    print(f"QTNs: {config['n_qtns']:,}")
    print(f"Chromosomes: {config['n_chromosomes']}")
    print(f"Traits: {config['n_traits']}")
    print(f"Population structure: {config['population_structure']['enabled']}")
    if config['population_structure']['enabled']:
        print(f"  Populations: {config['population_structure']['n_populations']}")
        print(f"  Sizes: {config['population_structure']['population_sizes']}")
        print(f"  FST: {config['population_structure']['fst']}")
    print(f"Missing data rate: {config['missing_data_rate']:.3f}")
    print(f"Random seed: {config['seed']}")
    print(f"Output directory: {config['output_dir']}")
    print()
    
    # Performance optimization info
    print(f"Performance optimizations:")
    print(f"  - Numba JIT compilation: {'✓ Available' if NUMBA_AVAILABLE else '✗ Not available (install numba for ~5-10x speedup)'}")
    print(f"  - Progress bars: {'✓ Available' if TQDM_AVAILABLE else '✗ Not available (install tqdm for progress tracking)'}")
    print(f"  - Vectorized operations: ✓ Enabled")
    print(f"  - Memory-efficient I/O: ✓ Enabled")
    print()

    # Initialize generator
    print("Initializing dataset generator...")
    generator = GWASDatasetGenerator(config)
    
    # Generate dataset components with timing
    start_time = time.time()
    
    print("Generating genetic map...")
    genetic_map = generator.generate_genetic_map()
    
    print("Generating genotype matrix with population structure...")
    genotypes, sample_ids = generator.generate_genotype_matrix(genetic_map)
    genotype_time = time.time() - start_time
    
    print("Selecting QTN positions and effects...")
    qtn_indices, qtn_effects = generator.select_qtns(genetic_map)
    
    print("Generating phenotypes with QTN effects...")
    pheno_start = time.time()
    phenotype_df = generator.generate_phenotypes(
        genotypes, qtn_indices, qtn_effects, sample_ids
    )
    phenotype_time = time.time() - pheno_start
    
    # Save complete dataset with timing
    save_start = time.time()
    output_dir = Path(config['output_dir'])
    generator.save_dataset(
        output_dir, genetic_map, genotypes, sample_ids,
        qtn_indices, qtn_effects, phenotype_df
    )
    save_time = time.time() - save_start
    total_time = time.time() - start_time
    
    print(f"\n=== Dataset Generation Summary ===")
    print(f"Generated {len(sample_ids):,} samples")
    print(f"Generated {len(genetic_map):,} SNPs across {config['n_chromosomes']} chromosomes")
    print(f"Selected {len(qtn_indices)} QTNs with effects ranging from {np.min(qtn_effects):.3f} to {np.max(qtn_effects):.3f}")
    print(f"Generated {config['n_traits']} correlated traits")
    if hasattr(generator, 'realized_heritability'):
        avg_h2 = np.mean(generator.realized_heritability)
        print(f"Realized heritability: {avg_h2:.3f} (target: {config['heritability']})")
    if config['population_structure']['enabled']:
        print(f"Simulated {config['population_structure']['n_populations']} populations with FST = {config['population_structure']['fst']}")
    print(f"Missing data rate: {np.mean(genotypes == -9):.4f}")

    # MAF-effect coupling validation
    if len(qtn_indices) > 0:
        qtn_mafs = generator.ancestral_freqs[qtn_indices]
        maf_effect_corr = np.corrcoef(qtn_mafs, np.abs(qtn_effects))[0, 1]
        print(f"MAF-effect correlation: {maf_effect_corr:.3f}")
    
    print(f"\n=== Performance Summary ===")
    print(f"Genotype generation: {genotype_time:.1f}s")
    print(f"Phenotype generation: {phenotype_time:.1f}s")
    print(f"File I/O operations: {save_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")
    
    # Performance breakdown
    compute_time = genotype_time + phenotype_time
    print(f"\nBreakdown:")
    print(f"  Computation: {compute_time:.1f}s ({100*compute_time/total_time:.1f}%)")
    print(f"  File I/O: {save_time:.1f}s ({100*save_time/total_time:.1f}%)")
    
    print(f"\nDataset saved to: {output_dir.absolute()}")
    print("Files created:")
    file_list = ['dataset_summary.json', 'design_summary.tsv', 'truth.tsv', 'genotypes.npz', 'map.csv',
                 'phenotype.csv', 'phenotype_null.csv', 'sample_names.csv', 'snp_names.csv', 'true_qtns.csv']
    # Check for both parquet and csv genotype files
    if (output_dir / 'genotype_numeric.parquet').exists():
        file_list.append('genotype_numeric.parquet')
    elif (output_dir / 'genotype_numeric.csv').exists():
        file_list.append('genotype_numeric.csv')

    for file in file_list:
        file_path = output_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  - {file}: {size_mb:.1f} MB")

if __name__ == '__main__':
    main()