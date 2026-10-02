"""Locus identification by LD clumping with in-sample r^2.

Follows PLINK ``--clump``: significant markers are taken in order of
increasing p-value; each unclaimed index marker claims every marker within
``clump_kb`` that has ``p < p_secondary`` and ``r^2 >= clump_r2`` with it.
The locus spans its claimed markers, and loci whose spans overlap (or lie
within ``merge_gap_kb``) on the same chromosome are merged, keeping the
strongest lead. r^2 comes from the analysed genotypes, so no reference panel
is needed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..matrix.ld import pairwise_r2
from ..utils.data_types import GenotypeMatrix

LOCUS_COLUMNS = [
    "Locus", "CHROM", "Start", "End", "Lead_SNP", "Lead_POS", "Lead_P",
    "Lead_Effect", "N_Significant", "N_Clumped", "Clumped_SNPs",
]


def identify_loci(
    pvalues: np.ndarray,
    chromosomes: Sequence,
    positions: Sequence,
    genotype: Union[GenotypeMatrix, np.ndarray],
    *,
    p_threshold: float,
    p_secondary: Optional[float] = None,
    clump_kb: float = 250.0,
    clump_r2: float = 0.1,
    merge_gap_kb: float = 0.0,
    marker_ids: Optional[Sequence] = None,
    effects: Optional[np.ndarray] = None,
    max_listed_snps: int = 50,
) -> pd.DataFrame:
    """Group significant markers into independent loci.

    Args:
        pvalues: per-marker p-values aligned with ``genotype`` columns.
        chromosomes, positions: marker map.
        genotype: genotypes used for in-sample r^2.
        p_threshold: index-marker significance threshold.
        p_secondary: threshold for markers that may join a clump
            (default: ``p_threshold``).
        clump_kb: maximum distance from the index marker.
        clump_r2: minimum r^2 with the index marker to join its clump.
        merge_gap_kb: merge loci on the same chromosome closer than this.
        marker_ids, effects: optional annotations for the output.
    """
    p = np.asarray(pvalues, dtype=float)
    chrom = np.asarray(chromosomes).astype(str)
    pos = np.asarray(positions, dtype=np.float64)
    if p_secondary is None:
        p_secondary = p_threshold
    p_secondary = max(p_secondary, p_threshold)
    ids = np.asarray(marker_ids).astype(str) if marker_ids is not None else np.array(
        [f"{c}:{int(x)}" for c, x in zip(chrom, pos)]
    )
    valid = np.isfinite(p)
    index_candidates = np.flatnonzero(valid & (p <= p_threshold))
    if index_candidates.size == 0:
        return pd.DataFrame(columns=LOCUS_COLUMNS)
    secondary = valid & (p <= p_secondary)
    claimed = np.zeros(p.size, dtype=bool)
    window = clump_kb * 1000.0

    clumps = []
    for idx in index_candidates[np.argsort(p[index_candidates], kind="stable")]:
        if claimed[idx]:
            continue
        near = np.flatnonzero(
            secondary & ~claimed & (chrom == chrom[idx]) & (np.abs(pos - pos[idx]) <= window)
        )
        near = near[near != idx]
        members = [int(idx)]
        if near.size:
            r2 = pairwise_r2(genotype, int(idx), near)
            members.extend(int(j) for j in near[r2 >= clump_r2])
        members = np.asarray(members, dtype=np.int64)
        claimed[members] = True
        clumps.append({
            "chrom": chrom[idx],
            "start": float(pos[members].min()),
            "end": float(pos[members].max()),
            "lead": int(idx),
            "members": members,
        })

    # Merge clumps whose spans overlap / are within merge_gap_kb.
    clumps.sort(key=lambda c: (c["chrom"], c["start"]))
    merged = []
    gap = merge_gap_kb * 1000.0
    for c in clumps:
        if merged and merged[-1]["chrom"] == c["chrom"] and c["start"] <= merged[-1]["end"] + gap:
            last = merged[-1]
            last["end"] = max(last["end"], c["end"])
            last["members"] = np.concatenate([last["members"], c["members"]])
            if p[c["lead"]] < p[last["lead"]]:
                last["lead"] = c["lead"]
        else:
            merged.append(dict(c))

    merged.sort(key=lambda c: p[c["lead"]])
    rows = []
    for k, c in enumerate(merged, start=1):
        members = np.unique(c["members"])
        members = members[np.argsort(p[members], kind="stable")]
        lead = c["lead"]
        rows.append({
            "Locus": k,
            "CHROM": c["chrom"],
            "Start": int(c["start"]),
            "End": int(c["end"]),
            "Lead_SNP": ids[lead],
            "Lead_POS": int(pos[lead]),
            "Lead_P": float(p[lead]),
            "Lead_Effect": float(effects[lead]) if effects is not None else np.nan,
            "N_Significant": int(np.sum(p[members] <= p_threshold)),
            "N_Clumped": int(members.size),
            "Clumped_SNPs": ";".join(ids[members[:max_listed_snps]]),
        })
    return pd.DataFrame(rows, columns=LOCUS_COLUMNS)
