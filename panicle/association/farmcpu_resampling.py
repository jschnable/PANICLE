"""FarmCPU resampling wrapper to stabilize marker discovery via RMIP"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.data_types import LEGACY_MARKER_ID_COLUMN, MARKER_ID_COLUMN, GenotypeMap, GenotypeMatrix
from .farmcpu import PANICLE_FarmCPU


@dataclass
class FarmCPUResamplingEntry:
    """Single output row for FarmCPU resampling results."""

    marker_index: int
    snp: str
    chrom: Union[str, int]
    pos: Union[int, float]
    rmip: float
    cluster_size: int = 1
    cluster_members: Optional[Dict[str, float]] = None

    def to_row(self, trait_name: str, include_cluster_details: bool) -> Dict[str, Union[str, int, float]]:
        row = {
            MARKER_ID_COLUMN: self.snp,
            LEGACY_MARKER_ID_COLUMN: self.snp,
            "Chr": self.chrom,
            "Pos": self.pos,
            "RMIP": float(self.rmip),
            "Trait": trait_name,
        }
        if include_cluster_details:
            if self.cluster_members is None:
                row["ClusterMembers"] = "size=1"
            else:
                members = ", ".join(
                    f"{marker}:{rmip:.3f}" for marker, rmip in self.cluster_members.items()
                )
                row["ClusterMembers"] = f"size={self.cluster_size}; {members}"
        return row


class FarmCPUResamplingResults:
    """Container for FarmCPU resampling outputs including RMIP values."""

    def __init__(
        self,
        entries: Sequence[FarmCPUResamplingEntry],
        trait_name: str,
        total_runs: int,
        cluster_mode: bool,
        per_marker_counts: Optional[Dict[int, int]] = None,
    ) -> None:
        self.entries = list(entries)
        self.trait_name = trait_name
        self.total_runs = total_runs
        self.cluster_mode = cluster_mode
        self.per_marker_counts = per_marker_counts or {}
        self.is_farmcpu_resampling = True

    @property
    def rmip_values(self) -> np.ndarray:
        return np.array([entry.rmip for entry in self.entries], dtype=np.float64)

    @property
    def snp_labels(self) -> List[str]:
        return [entry.snp for entry in self.entries]

    @property
    def chromosomes(self) -> np.ndarray:
        return np.array([entry.chrom for entry in self.entries])

    @property
    def positions(self) -> np.ndarray:
        return np.array([entry.pos for entry in self.entries], dtype=np.float64)

    def to_dataframe(self) -> pd.DataFrame:
        include_cluster_details = self.cluster_mode
        rows = [entry.to_row(self.trait_name, include_cluster_details) for entry in self.entries]
        if not rows:
            columns = [MARKER_ID_COLUMN, LEGACY_MARKER_ID_COLUMN, "Chr", "Pos", "RMIP", "Trait"]
            if include_cluster_details:
                columns.append("ClusterMembers")
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows)

    def to_numpy(self) -> np.ndarray:
        """Provide a numpy view for compatibility with downstream utilities."""

        if not self.entries:
            return np.zeros((0, 3), dtype=np.float64)
        rmip = self.rmip_values
        nan_array = np.full_like(rmip, np.nan)
        return np.column_stack([nan_array, nan_array, rmip])


def _ensure_map_arrays(map_data: GenotypeMap) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    snp_ids = map_data.marker_ids.values
    chroms = map_data.chromosomes.values
    positions = map_data.positions.values
    return snp_ids, chroms, positions


def _validate_inputs(runs: int, mask_proportion: float, ld_threshold: float) -> None:
    if runs <= 0:
        raise ValueError("Number of FarmCPU runs must be positive")
    if not (0.0 <= mask_proportion < 1.0):
        raise ValueError("Mask proportion must be in [0, 1)")
    if ld_threshold < 0.0 or ld_threshold > 1.0:
        raise ValueError("LD threshold must be between 0 and 1 inclusive")


def _get_valid_trait_indices(trait_values: np.ndarray) -> np.ndarray:
    return np.where(~np.isnan(trait_values))[0]


def _subset_covariates(cv: Optional[np.ndarray], indices: np.ndarray) -> Optional[np.ndarray]:
    if cv is None:
        return None
    return cv[indices]


def _fetch_imputed_genotypes(
    genotype: Union[GenotypeMatrix, np.ndarray],
    marker_indices: Sequence[int],
) -> np.ndarray:
    if isinstance(genotype, GenotypeMatrix):
        return genotype.get_columns_imputed(marker_indices)

    matrix = np.asarray(genotype)[:, marker_indices].astype(np.float64)
    missing_mask = (matrix == -9) | np.isnan(matrix)
    if not missing_mask.any():
        return matrix

    for col_idx in range(matrix.shape[1]):
        column = matrix[:, col_idx]
        mask = missing_mask[:, col_idx]
        if not mask.any():
            continue
        valid_values = column[~mask]
        if valid_values.size == 0:
            fill_value = 0.0
        else:
            counts = np.bincount(valid_values.astype(int), minlength=3)
            fill_value = float(np.argmax(counts))
        column[mask] = fill_value
        matrix[:, col_idx] = column
    return matrix


def _build_clusters(
    marker_counts: np.ndarray,
    run_marker_sets: Sequence[set],
    genotype: Union[GenotypeMatrix, np.ndarray],
    snp_ids: np.ndarray,
    chroms: np.ndarray,
    positions: np.ndarray,
    total_runs: int,
    ld_threshold: float,
) -> List[FarmCPUResamplingEntry]:
    identified_mask = marker_counts > 0
    identified_indices = np.where(identified_mask)[0]
    if identified_indices.size == 0:
        return []

    imputed = _fetch_imputed_genotypes(genotype, identified_indices)
    if imputed.shape[1] == 1:
        idx = identified_indices[0]
        rmip = marker_counts[idx] / total_runs
        return [
            FarmCPUResamplingEntry(
                marker_index=int(idx),
                snp=str(snp_ids[idx]),
                chrom=chroms[idx],
                pos=positions[idx],
                rmip=float(rmip),
                cluster_size=1,
                cluster_members={str(snp_ids[idx]): float(rmip)},
            )
        ]

    correlation = np.corrcoef(imputed, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    r2_matrix = correlation ** 2
    adjacency = r2_matrix >= ld_threshold
    np.fill_diagonal(adjacency, True)

    visited = set()
    clusters: List[List[int]] = []
    for idx in range(len(identified_indices)):
        if idx in visited:
            continue
        queue = [idx]
        component = []
        visited.add(idx)
        while queue:
            node = queue.pop()
            component.append(node)
            neighbors = np.where(adjacency[node])[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        clusters.append(component)

    entries: List[FarmCPUResamplingEntry] = []
    for component in clusters:
        component_marker_indices = identified_indices[np.array(component, dtype=int)]
        per_marker_counts = marker_counts[component_marker_indices]
        representative_idx = int(component_marker_indices[np.argmax(per_marker_counts)])

        cluster_markers_set = set(int(idx) for idx in component_marker_indices)
        run_hits = 0
        for run_markers in run_marker_sets:
            if run_markers & cluster_markers_set:
                run_hits += 1
        rmip_cluster = run_hits / total_runs if total_runs else 0.0

        member_rmip = {
            str(snp_ids[idx]): marker_counts[idx] / total_runs for idx in component_marker_indices
        }

        entries.append(
            FarmCPUResamplingEntry(
                marker_index=representative_idx,
                snp=str(snp_ids[representative_idx]),
                chrom=chroms[representative_idx],
                pos=positions[representative_idx],
                rmip=float(rmip_cluster),
                cluster_size=len(component_marker_indices),
                cluster_members=member_rmip,
            )
        )

    entries.sort(key=lambda entry: entry.rmip, reverse=True)
    return entries


def _build_non_cluster_entries(
    marker_counts: np.ndarray,
    snp_ids: np.ndarray,
    chroms: np.ndarray,
    positions: np.ndarray,
    total_runs: int,
) -> List[FarmCPUResamplingEntry]:
    identified_indices = np.where(marker_counts > 0)[0]
    entries: List[FarmCPUResamplingEntry] = []
    for idx in identified_indices:
        rmip_value = marker_counts[idx] / total_runs if total_runs else 0.0
        entries.append(
            FarmCPUResamplingEntry(
                marker_index=int(idx),
                snp=str(snp_ids[idx]),
                chrom=chroms[idx],
                pos=positions[idx],
                rmip=float(rmip_value),
                cluster_size=1,
                cluster_members={str(snp_ids[idx]): float(rmip_value)},
            )
        )

    entries.sort(key=lambda entry: entry.rmip, reverse=True)
    return entries


def PANICLE_FarmCPUResampling(
    phe: np.ndarray,
    geno: Union[GenotypeMatrix, np.ndarray],
    map_data: GenotypeMap,
    CV: Optional[np.ndarray] = None,
    *,
    runs: int = 100,
    mask_proportion: float = 0.1,
    significance_threshold: float = 5e-8,
    cluster_markers: bool = False,
    ld_threshold: float = 0.7,
    trait_name: str = "Trait",
    random_seed: Optional[int] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    **farmcpu_kwargs,
) -> FarmCPUResamplingResults:
    """Run FarmCPU repeatedly with phenotype masking to calculate RMIP."""

    _validate_inputs(runs, mask_proportion, ld_threshold)

    if phe.ndim != 2 or phe.shape[1] != 2:
        raise ValueError("Phenotype matrix must have shape (n_individuals, 2)")

    trait_values = phe[:, 1].astype(np.float64)
    valid_trait_indices = _get_valid_trait_indices(trait_values)
    total_individuals = phe.shape[0]

    if valid_trait_indices.size == 0:
        raise ValueError("No valid (non-missing) phenotype values available")

    if isinstance(geno, GenotypeMatrix):
        n_markers = geno.n_markers
    else:
        geno = np.asarray(geno)
        if geno.ndim != 2:
            raise ValueError("Genotype must be a 2D array")
        n_markers = geno.shape[1]

    marker_counts = np.zeros(n_markers, dtype=np.int32)
    run_marker_lists: List[np.ndarray] = []

    rng = np.random.default_rng(random_seed)

    snp_ids, chroms, positions = _ensure_map_arrays(map_data)

    for run_idx in range(runs):
        run_start = time.time()
        mask_count = int(round(mask_proportion * valid_trait_indices.size))
        if mask_count == 0 and mask_proportion > 0.0 and valid_trait_indices.size > 1:
            mask_count = 1
        if mask_count >= valid_trait_indices.size:
            mask_count = max(valid_trait_indices.size - 1, 0)

        if mask_count > 0:
            masked = rng.choice(valid_trait_indices, size=mask_count, replace=False)
            keep_indices = np.setdiff1d(valid_trait_indices, masked, assume_unique=True)
        else:
            keep_indices = valid_trait_indices.copy()

        if keep_indices.size == 0:
            raise ValueError("All individuals were masked; decrease mask_proportion")

        phe_run = phe[keep_indices]
        if isinstance(geno, GenotypeMatrix):
            geno_run = geno[keep_indices, :]
        else:
            geno_run = geno[keep_indices, :]
        cv_run = _subset_covariates(CV, keep_indices)

        farmcpu_result = PANICLE_FarmCPU(
            phe=phe_run,
            geno=geno_run,
            map_data=map_data,
            CV=cv_run,
            verbose=verbose,
            **farmcpu_kwargs,
        )

        pvalues = farmcpu_result.pvalues
        significant_markers = np.where(pvalues <= significance_threshold)[0]
        marker_counts[significant_markers] += 1
        run_marker_lists.append(significant_markers.astype(np.int32))
        if progress_callback is not None:
            progress_callback(run_idx + 1, runs, time.time() - run_start)

    if cluster_markers:
        run_marker_sets = [set(marker_list.tolist()) for marker_list in run_marker_lists]
        entries = _build_clusters(
            marker_counts=marker_counts,
            run_marker_sets=run_marker_sets,
            genotype=geno,
            snp_ids=snp_ids,
            chroms=chroms,
            positions=positions,
            total_runs=runs,
            ld_threshold=ld_threshold,
        )
    else:
        entries = _build_non_cluster_entries(
            marker_counts=marker_counts,
            snp_ids=snp_ids,
            chroms=chroms,
            positions=positions,
            total_runs=runs,
        )

    return FarmCPUResamplingResults(
        entries=entries,
        trait_name=trait_name,
        total_runs=runs,
        cluster_mode=cluster_markers,
        per_marker_counts={idx: int(count) for idx, count in enumerate(marker_counts) if count > 0},
    )
