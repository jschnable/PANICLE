#!/usr/bin/env python3
"""Deterministic MLM performance guardrail benchmark.

This script provides a lightweight benchmark for PANICLE MLM paths and can
optionally compare current metrics against a saved baseline JSON.

Default behavior is non-blocking: regressions emit warnings but exit 0.
Use --strict to fail on regression.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from panicle.association.mlm import PANICLE_MLM
from panicle.association.mlm_loco import PANICLE_MLM_LOCO
from panicle.data.loaders import load_genotype_file
from panicle.matrix.kinship import PANICLE_K_VanRaden
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.utils.data_types import GenotypeMatrix


def _peak_rss_mb() -> float:
    """Return process peak RSS in MB."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(ru.ru_maxrss)
    # macOS reports bytes, Linux reports KB.
    if platform.system() == "Darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _build_synthetic_dataset(
    seed: int,
    n_individuals: int,
    n_markers: int,
    n_covariates: int,
    n_chromosomes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    geno = rng.integers(0, 3, size=(n_individuals, n_markers), dtype=np.int8)

    # Inject deterministic missingness to exercise imputation code paths.
    missing_mask = rng.random((n_individuals, n_markers)) < 0.02
    geno_missing = geno.copy()
    geno_missing[missing_mask] = -9

    cov = rng.normal(size=(n_individuals, n_covariates)).astype(np.float64)
    y = (cov @ rng.normal(size=n_covariates)) + rng.normal(size=n_individuals)
    phe = np.column_stack([np.arange(n_individuals), y]).astype(np.float64)

    chrom_blocks = np.repeat(np.arange(1, n_chromosomes + 1), int(np.ceil(n_markers / n_chromosomes)))[:n_markers]
    map_df = pd.DataFrame(
        {
            "SNP": [f"SNP{i:07d}" for i in range(n_markers)],
            "CHROM": chrom_blocks.astype(str),
            "POS": np.arange(1, n_markers + 1, dtype=np.int64),
        }
    )

    return geno_missing, phe, cov, map_df


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    if args.genotype_file:
        load_t0 = time.perf_counter()
        geno_matrix, _, geno_map = load_genotype_file(
            args.genotype_file,
            file_format=args.genotype_format,
        )
        load_seconds = time.perf_counter() - load_t0
        if geno_map is None:
            raise ValueError("Real-data mode requires genotype map information")
        map_df = geno_map.to_dataframe()

        n_total = geno_matrix.n_markers
        if args.max_markers is not None and 0 < args.max_markers < n_total:
            if args.contiguous_markers:
                idx = np.arange(args.max_markers, dtype=int)
            else:
                idx = np.linspace(0, n_total - 1, args.max_markers, dtype=int)
            geno = GenotypeMatrix(
                geno_matrix._data[:, idx],
                is_imputed=geno_matrix.is_imputed,
                precompute_alleles=not geno_matrix.is_imputed,
            )
            map_df = map_df.iloc[idx].reset_index(drop=True)
        else:
            geno = geno_matrix

        rng = np.random.default_rng(args.seed)
        n = geno.n_individuals
        cov = rng.normal(size=(n, args.n_covariates)).astype(np.float64)
        y = (cov @ rng.normal(size=args.n_covariates)) + rng.normal(size=n)
        phe = np.column_stack([np.arange(n), y]).astype(np.float64)
        data_mode = "real"
    else:
        geno, phe, cov, map_df = _build_synthetic_dataset(
            seed=args.seed,
            n_individuals=args.n_individuals,
            n_markers=args.n_markers,
            n_covariates=args.n_covariates,
            n_chromosomes=args.n_chromosomes,
        )
        load_seconds = 0.0
        data_mode = "synthetic"

    metrics: Dict[str, Any] = {
        "config": {
            "data_mode": data_mode,
            "seed": args.seed,
            "n_individuals": int(geno.shape[0] if isinstance(geno, np.ndarray) else geno.n_individuals),
            "n_markers": int(geno.shape[1] if isinstance(geno, np.ndarray) else geno.n_markers),
            "n_covariates": args.n_covariates,
            "n_chromosomes": args.n_chromosomes,
            "cpu": args.cpu,
            "maxLine": args.maxline,
            "genotype_file": args.genotype_file,
            "genotype_format": args.genotype_format,
            "max_markers": args.max_markers,
            "contiguous_markers": bool(args.contiguous_markers),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "metrics": {},
    }

    metrics["metrics"]["load_time_sec"] = float(load_seconds)

    t0 = time.perf_counter()
    kinship = PANICLE_K_VanRaden(geno, maxLine=args.maxline, verbose=False)
    metrics["metrics"]["kinship_time_sec"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    mlm_res = PANICLE_MLM(
        phe=phe,
        geno=geno,
        K=kinship,
        CV=cov,
        maxLine=args.maxline,
        cpu=args.cpu,
        verbose=False,
    )
    metrics["metrics"]["mlm_time_sec"] = time.perf_counter() - t1
    metrics["metrics"]["mlm_min_p"] = float(np.nanmin(mlm_res.pvalues))

    t2 = time.perf_counter()
    loco = PANICLE_K_VanRaden_LOCO(geno, map_df, maxLine=args.maxline, cpu=args.cpu, verbose=False)
    metrics["metrics"]["loco_kinship_time_sec"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    mlm_loco_res = PANICLE_MLM_LOCO(
        phe=phe,
        geno=geno,
        map_data=map_df,
        loco_kinship=loco,
        CV=cov,
        maxLine=args.maxline,
        cpu=args.cpu,
        lrt_refinement=False,
        verbose=False,
    )
    metrics["metrics"]["mlm_loco_time_sec"] = time.perf_counter() - t3
    metrics["metrics"]["mlm_loco_min_p"] = float(np.nanmin(mlm_loco_res.pvalues))
    metrics["metrics"]["mlm_loco_candidates_p_lt_1e4"] = int(np.sum(mlm_loco_res.pvalues < 1e-4))

    metrics["metrics"]["peak_rss_mb"] = _peak_rss_mb()
    return metrics


def _compare_to_baseline(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    warn_threshold: float,
) -> List[str]:
    issues: List[str] = []
    keys = [
        "kinship_time_sec",
        "mlm_time_sec",
        "loco_kinship_time_sec",
        "mlm_loco_time_sec",
        "peak_rss_mb",
    ]

    cur_metrics = current.get("metrics", {})
    base_metrics = baseline.get("metrics", {})

    for key in keys:
        cur = cur_metrics.get(key)
        base = base_metrics.get(key)
        if cur is None or base is None:
            continue
        if not np.isfinite(base) or base <= 0:
            continue
        ratio = float(cur) / float(base)
        if ratio > (1.0 + warn_threshold):
            issues.append(
                f"{key}: current={cur:.4f}, baseline={base:.4f}, ratio={ratio:.3f}"
            )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MLM perf benchmark and optional regression check")
    parser.add_argument("--seed", type=int, default=20260216)
    parser.add_argument("--n-individuals", type=int, default=220)
    parser.add_argument("--n-markers", type=int, default=120000)
    parser.add_argument("--n-covariates", type=int, default=3)
    parser.add_argument("--n-chromosomes", type=int, default=12)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--maxline", type=int, default=2000)
    parser.add_argument("--genotype-file", type=str, default=None, help="Optional real genotype file path")
    parser.add_argument("--genotype-format", type=str, default=None, help="Optional genotype format override")
    parser.add_argument("--max-markers", type=int, default=None, help="Optional marker subsample size in real-data mode")
    parser.add_argument("--contiguous-markers", action="store_true", help="Use first max_markers instead of evenly spaced selection")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline JSON for regression check")
    parser.add_argument("--write-baseline", type=Path, default=None, help="Write current metrics as baseline JSON")
    parser.add_argument("--warn-threshold", type=float, default=0.20, help="Relative slowdown threshold (default 0.20 = 20%%)")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when regression threshold is exceeded")
    args = parser.parse_args()

    report = run_benchmark(args)
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.write_baseline is not None:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote baseline: {args.write_baseline}")

    issues: List[str] = []
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        issues = _compare_to_baseline(report, baseline, warn_threshold=args.warn_threshold)
        if issues:
            print("\nPotential performance regressions detected:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nNo performance regressions detected at configured threshold.")

    if issues and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
