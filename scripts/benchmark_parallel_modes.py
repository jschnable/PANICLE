#!/usr/bin/env python3
"""Benchmark MLM runtime across GWASPipeline parallel mode settings.

Primary comparison:
1) parallel_mode='off', ncpus=0  -> resolves to single-thread method engines
2) parallel_mode='auto', ncpus=0 -> resolves to all available CPU cores

The benchmark measures setup and analysis phases separately and reports
replicate statistics plus estimated speedup.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Allow running directly from repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panicle.data.loaders import (
    detect_file_format,
    load_covariate_file,
    load_genotype_file,
    load_map_file,
    load_phenotype_file,
)
from panicle.pipelines.gwas import GWASPipeline, _resolve_method_cpu
from panicle.utils.data_types import GenotypeMap, GenotypeMatrix


def _build_synthetic_inputs(
    *,
    seed: int,
    n_individuals: int,
    n_markers: int,
    n_chromosomes: int,
    n_covariates: int,
    trait_name: str,
    n_causal_markers: int,
    causal_effect: float,
    missing_rate: float,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    ids = [f"id{i:07d}" for i in range(n_individuals)]

    geno = rng.integers(0, 3, size=(n_individuals, n_markers), dtype=np.int8)
    if missing_rate > 0.0:
        miss = rng.random((n_individuals, n_markers)) < missing_rate
        geno = geno.copy()
        geno[miss] = -9

    cov = rng.normal(size=(n_individuals, n_covariates)).astype(np.float64)
    cov_beta = rng.normal(scale=0.25, size=n_covariates)
    trait = cov @ cov_beta + rng.normal(scale=1.0, size=n_individuals)

    n_causal = max(1, min(n_causal_markers, n_markers))
    causal_idx = np.sort(rng.choice(n_markers, size=n_causal, replace=False))
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_causal)
    effect_sizes = signs * causal_effect
    trait = trait + geno[:, causal_idx].astype(np.float64) @ effect_sizes

    phenotype_df = pd.DataFrame({"ID": ids, trait_name: trait})
    covariate_df = pd.DataFrame({"ID": ids})
    for i in range(n_covariates):
        covariate_df[f"Cov{i+1}"] = cov[:, i]

    chrom = np.repeat(np.arange(1, n_chromosomes + 1), int(np.ceil(n_markers / n_chromosomes)))[:n_markers]
    map_df = pd.DataFrame(
        {
            "SNP": [f"SNP{i:07d}" for i in range(n_markers)],
            "CHROM": [f"Chr{int(c):02d}" for c in chrom],
            "POS": np.arange(1, n_markers + 1, dtype=np.int64),
        }
    )

    return {
        "phenotype_df": phenotype_df,
        "genotype_matrix": GenotypeMatrix(geno),
        "geno_map": GenotypeMap(map_df),
        "individual_ids": ids,
        "covariate_df": covariate_df,
        "covariate_names": [f"Cov{i+1}" for i in range(n_covariates)],
        "mode": "synthetic",
    }


def _load_real_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    if args.phenotype is None or args.genotype is None:
        raise ValueError("Real-data mode requires --phenotype and --genotype")

    fmt = args.genotype_format
    if fmt is None:
        fmt = detect_file_format(args.genotype)

    phe = load_phenotype_file(
        args.phenotype,
        trait_columns=[args.trait],
        id_column=args.phenotype_id_column,
    )
    geno, individual_ids, geno_map = load_genotype_file(args.genotype, file_format=fmt)

    if args.map is not None:
        supplied_map = load_map_file(args.map)
        if supplied_map.n_markers != geno_map.n_markers:
            raise ValueError(
                f"Map marker count ({supplied_map.n_markers}) != genotype marker count ({geno_map.n_markers})"
            )
        geno_map = supplied_map

    covariate_df = None
    covariate_names: List[str] = []
    if args.covariates is not None:
        cov_cols = [c.strip() for c in args.covariate_columns.split(",") if c.strip()] if args.covariate_columns else None
        covariate_df = load_covariate_file(
            args.covariates,
            covariate_columns=cov_cols,
            id_column=args.covariate_id_column,
        )
        covariate_names = [c for c in covariate_df.columns if c != "ID"]

    return {
        "phenotype_df": phe,
        "genotype_matrix": geno,
        "geno_map": geno_map,
        "individual_ids": list(individual_ids),
        "covariate_df": covariate_df,
        "covariate_names": covariate_names,
        "mode": "real",
    }


def _new_pipeline_from_inputs(inputs: Dict[str, Any], out_dir: Path) -> GWASPipeline:
    p = GWASPipeline(output_dir=out_dir)
    p.phenotype_df = inputs["phenotype_df"].copy()
    p.genotype_matrix = inputs["genotype_matrix"]
    p.geno_map = inputs["geno_map"]
    p.individual_ids = list(inputs["individual_ids"])
    if inputs["covariate_df"] is not None:
        p.covariate_df = inputs["covariate_df"].copy()
        p.covariate_names = list(inputs["covariate_names"])
    return p


def _run_once(
    *,
    inputs: Dict[str, Any],
    trait: str,
    n_pcs: int,
    ncpus: int,
    parallel_mode: str,
    max_iterations: int,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _new_pipeline_from_inputs(inputs, output_dir)
    pipeline.log = lambda *_args, **_kwargs: None  # keep benchmark output concise

    t0 = time.perf_counter()
    pipeline.align_samples()
    t1 = time.perf_counter()
    # Precompute kinship once in setup to avoid run_analysis recomputing it.
    pipeline.compute_population_structure(n_pcs=n_pcs, calculate_kinship=True)
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    pipeline.run_analysis(
        traits=[trait],
        methods=["MLM"],
        max_iterations=max_iterations,
        ncpus=ncpus,
        parallel_mode=parallel_mode,
        outputs=["all_marker_pvalues"],
    )
    t4 = time.perf_counter()

    result_file = output_dir / f"GWAS_{trait}_all_results.csv"
    n_top = None
    if result_file.exists():
        df = pd.read_csv(result_file)
        if "MLM_P" in df.columns:
            n_top = int(np.sum(df["MLM_P"].to_numpy(dtype=float) < 1e-4))

    return {
        "parallel_mode": parallel_mode,
        "ncpus_requested": int(ncpus),
        "ncpus_effective": int(_resolve_method_cpu(ncpus=ncpus, parallel_mode=parallel_mode)),
        "align_seconds": float(t1 - t0),
        "structure_seconds": float(t2 - t1),
        "analysis_seconds": float(t4 - t3),
        "total_seconds": float(t4 - t0),
        "n_sig_p_lt_1e4": n_top,
        "output_dir": str(output_dir),
    }


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["parallel_mode"], []).append(row)

    summary: Dict[str, Any] = {"modes": {}}
    for mode, mode_rows in by_mode.items():
        analysis = np.array([r["analysis_seconds"] for r in mode_rows], dtype=float)
        total = np.array([r["total_seconds"] for r in mode_rows], dtype=float)
        summary["modes"][mode] = {
            "runs": len(mode_rows),
            "analysis_mean_sec": float(np.mean(analysis)),
            "analysis_median_sec": float(np.median(analysis)),
            "analysis_std_sec": float(np.std(analysis)),
            "total_mean_sec": float(np.mean(total)),
            "total_median_sec": float(np.median(total)),
            "effective_cpus": int(mode_rows[0]["ncpus_effective"]),
        }

    if "off" in summary["modes"] and "auto" in summary["modes"]:
        off_med = summary["modes"]["off"]["analysis_median_sec"]
        auto_med = summary["modes"]["auto"]["analysis_median_sec"]
        if auto_med > 0:
            summary["analysis_speedup_off_to_auto_x"] = float(off_med / auto_med)
            summary["analysis_time_reduction_pct"] = float(100.0 * (off_med - auto_med) / off_med)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GWASPipeline MLM runtime for parallel_mode off vs auto",
    )

    # Mode selection.
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic in-memory dataset")

    # Real data inputs.
    parser.add_argument("--phenotype", type=str, default=None)
    parser.add_argument("--phenotype-id-column", type=str, default="ID")
    parser.add_argument("--genotype", type=str, default=None)
    parser.add_argument("--genotype-format", type=str, default=None)
    parser.add_argument("--map", type=str, default=None)
    parser.add_argument("--covariates", type=str, default=None)
    parser.add_argument("--covariate-columns", type=str, default=None)
    parser.add_argument("--covariate-id-column", type=str, default="ID")
    parser.add_argument("--trait", type=str, default="Trait1")

    # Synthetic controls.
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument("--n-individuals", type=int, default=240)
    parser.add_argument("--n-markers", type=int, default=80000)
    parser.add_argument("--n-chromosomes", type=int, default=12)
    parser.add_argument("--n-covariates", type=int, default=3)
    parser.add_argument("--n-causal-markers", type=int, default=40)
    parser.add_argument("--causal-effect", type=float, default=0.25)
    parser.add_argument("--missing-rate", type=float, default=0.02)

    # Benchmark controls.
    parser.add_argument("--ncpus", type=int, default=0, help="CPU request passed into run_analysis")
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per mode")
    parser.add_argument("--repeats", type=int, default=3, help="Measured runs per mode")
    parser.add_argument("--output-root", type=Path, default=Path("scripts/perf_baselines/parallel_mode_runs"))
    parser.add_argument("--keep-run-artifacts", action="store_true", help="Keep GWAS output dirs for each run")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.synthetic:
        inputs = _build_synthetic_inputs(
            seed=args.seed,
            n_individuals=args.n_individuals,
            n_markers=args.n_markers,
            n_chromosomes=args.n_chromosomes,
            n_covariates=args.n_covariates,
            trait_name=args.trait,
            n_causal_markers=args.n_causal_markers,
            causal_effect=args.causal_effect,
            missing_rate=args.missing_rate,
        )
    else:
        inputs = _load_real_inputs(args)

    map_df = inputs["geno_map"].to_dataframe()
    n_markers = int(inputs["genotype_matrix"].n_markers)
    n_chrom = int(pd.Series(map_df["CHROM"]).astype(str).nunique())
    if n_markers < 50_000 or n_chrom < 3:
        print(
            "Warning: current workload may not trigger MLM-LOCO internal parallel path "
            f"(markers={n_markers}, chromosomes={n_chrom}; heuristic needs >=50,000 and >=3)."
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = args.output_root / f"parallel_mode_benchmark_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    mode_defs: List[Tuple[str, int]] = [("off", args.ncpus), ("auto", args.ncpus)]
    rows: List[Dict[str, Any]] = []

    # Warmups (per mode).
    for mode, ncpus in mode_defs:
        for warm_idx in range(args.warmup):
            label = f"warmup{warm_idx + 1:02d}"
            run_dir = root / f"{mode}_{label}"
            row = _run_once(
                inputs=inputs,
                trait=args.trait,
                n_pcs=args.n_pcs,
                ncpus=ncpus,
                parallel_mode=mode,
                max_iterations=args.max_iterations,
                output_dir=run_dir,
            )
            row["phase"] = "warmup"
            row["replicate"] = int(warm_idx + 1)
            rows.append(row)
            print(
                f"[{mode}:{label}] analysis={row['analysis_seconds']:.3f}s "
                f"total={row['total_seconds']:.3f}s cpus={row['ncpus_effective']}"
            )
            if not args.keep_run_artifacts:
                shutil.rmtree(run_dir, ignore_errors=True)

    # Measured runs: alternate mode order each round to reduce cache/order bias.
    per_mode_counts: Dict[str, int] = {mode: 0 for mode, _ in mode_defs}
    for round_idx in range(args.repeats):
        order = mode_defs if (round_idx % 2 == 0) else list(reversed(mode_defs))
        for mode, ncpus in order:
            per_mode_counts[mode] += 1
            label = f"run{per_mode_counts[mode]:02d}"
            run_dir = root / f"{mode}_{label}"
            row = _run_once(
                inputs=inputs,
                trait=args.trait,
                n_pcs=args.n_pcs,
                ncpus=ncpus,
                parallel_mode=mode,
                max_iterations=args.max_iterations,
                output_dir=run_dir,
            )
            row["phase"] = "measured"
            row["replicate"] = int(per_mode_counts[mode])
            row["round"] = int(round_idx + 1)
            rows.append(row)
            print(
                f"[{mode}:{label}] analysis={row['analysis_seconds']:.3f}s "
                f"total={row['total_seconds']:.3f}s cpus={row['ncpus_effective']}"
            )
            if not args.keep_run_artifacts:
                shutil.rmtree(run_dir, ignore_errors=True)

    measured_rows = [r for r in rows if r["phase"] == "measured"]
    summary = _summarize(measured_rows)

    report = {
        "config": {
            "synthetic": bool(args.synthetic),
            "trait": args.trait,
            "ncpus_requested": int(args.ncpus),
            "n_pcs": int(args.n_pcs),
            "max_iterations": int(args.max_iterations),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
        },
        "input": {
            "mode": inputs["mode"],
            "n_individuals": int(inputs["genotype_matrix"].n_individuals),
            "n_markers": n_markers,
            "n_chromosomes": n_chrom,
            "n_covariates": int(len(inputs["covariate_names"])),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": int(os.cpu_count() or 1),
        },
        "summary": summary,
        "runs": rows,
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    out_json = args.output_json
    if out_json is None:
        out_json = root / "benchmark_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote report: {out_json}")

    # If artifacts are not kept and no explicit output file was provided, root
    # only contains the report file; keep it for traceability.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
