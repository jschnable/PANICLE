"""Per-trait post-association step used by ``GWASPipeline.run_analysis``.

Adds to each method's summary row:

* ``H2_REML`` / ``H2_REML_SE`` - SNP heritability from the REML null model
  with the global (all-chromosome) kinship and the scan's fixed covariates;
  one value per trait, repeated on each method row and written with variance
  components to ``GWAS_<trait>_heritability.csv``;
* ``Lambda_1000`` - lambda_GC rescaled to 1000 samples (or 1000 cases +
  1000 controls for a binary trait);
* ``N_Loci`` - independent loci from in-sample LD clumping, written to
  ``GWAS_<trait>_<method>_loci.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from ..association.farmcpu_resampling import FarmCPUResamplingResults
from .heritability import REMLHeritability
from .inflation import case_control_counts, lambda_1000
from .loci import identify_loci


@dataclass
class PostGWASOptions:
    loci: bool = True
    heritability: bool = True
    # "auto": clumping window = in-sample LD decay distance (see GWASPipeline).
    clump_kb: Union[float, str] = "auto"
    clump_r2: float = 0.1
    clump_p_secondary: Optional[float] = None
    merge_gap_kb: float = 0.0

    @classmethod
    def from_dict(cls, loci: bool, heritability: bool,
                  params: Optional[Mapping] = None) -> "PostGWASOptions":
        params = dict(params or {})
        unknown = set(params) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown post-GWAS parameters: {sorted(unknown)}")
        opts = cls(loci=loci, heritability=heritability, **params)
        if isinstance(opts.clump_kb, str):
            if opts.clump_kb.strip().lower() != "auto":
                opts.clump_kb = float(opts.clump_kb)
            else:
                opts.clump_kb = "auto"
        return opts


def write_heritability(trait_name: str, h2: REMLHeritability, output_dir: Path) -> None:
    pd.DataFrame([{"Trait": trait_name, "Model": "REML null model, global kinship", **h2.to_dict()}]).to_csv(
        Path(output_dir) / f"GWAS_{trait_name}_heritability.csv", index=False,
    )


def run_post_gwas(
    *,
    trait_name: str,
    method_reports: Mapping,
    summary_rows: List[Dict],
    phenotype: np.ndarray,
    genotype,
    geno_map,
    output_dir: Path,
    options: PostGWASOptions,
    log: Callable[[str], None],
    heritability: Optional[REMLHeritability] = None,
) -> None:
    """Augment ``summary_rows`` in place and write locus tables."""
    n = int(np.isfinite(phenotype).sum())
    n_cases, n_controls = case_control_counts(phenotype)
    chrom = geno_map.chromosomes.to_numpy() if geno_map is not None else None
    pos = geno_map.positions.to_numpy() if geno_map is not None else None
    ids = geno_map.marker_ids.to_numpy() if geno_map is not None else None
    rows_by_method = {row.get("Method"): row for row in summary_rows if row.get("Trait") == trait_name}

    if heritability is not None:
        for row in rows_by_method.values():
            row["H2_REML"] = round(heritability.h2, 4)
            row["H2_REML_SE"] = round(heritability.h2_se, 4) if np.isfinite(heritability.h2_se) else np.nan

    for method, report in method_reports.items():
        res = report.run.result
        row = rows_by_method.get(method)
        if row is None or isinstance(res, FarmCPUResamplingResults):
            continue

        lam = report.run.lambda_gc
        if lam is not None:
            row["Lambda_1000"] = round(lambda_1000(lam, n, n_cases=n_cases, n_controls=n_controls), 4)
            log(f"   {method} Lambda_1000: {row['Lambda_1000']:.4f}")

        if options.loci and chrom is not None:
            loci = identify_loci(
                res.pvalues, chrom, pos, genotype,
                p_threshold=report.threshold.value,
                p_secondary=options.clump_p_secondary,
                clump_kb=options.clump_kb,
                clump_r2=options.clump_r2,
                merge_gap_kb=options.merge_gap_kb,
                marker_ids=ids,
                effects=res.effects,
            )
            row["N_Loci"] = int(len(loci))
            if len(loci):
                loci.insert(0, "Method", method)
                loci.insert(0, "Trait", trait_name)
                path = Path(output_dir) / f"GWAS_{trait_name}_{method}_loci.csv"
                loci.to_csv(path, index=False)
                log(f"   {method}: {len(loci)} independent loci -> {path.name}")
