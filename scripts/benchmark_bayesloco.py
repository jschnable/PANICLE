#!/usr/bin/env python3
"""Benchmark BAYESLOCO runtime, cost, and calibration on synthetic datasets.

This script runs BAYESLOCO across a grid of dataset sizes and reports:
1. Wall-clock and metadata timing breakdowns
2. Pass-equivalent cost accounting from BAYESLOCO metadata
3. Calibration diagnostics (lambda GC, null tail rates)
4. Power diagnostics (causal rank/contrast on synthetic signal traits)

Example:
  python scripts/benchmark_bayesloco.py \
    --sizes 120x2000,240x5000 \
    --replicates 2 \
    --scenarios null,power \
    --loco-modes subtract_only,refine \
    --output-json scripts/perf_baselines/bayesloco_benchmark_latest.json \
    --output-csv scripts/perf_baselines/bayesloco_benchmark_latest.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Add project root to import path when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panicle.association.bayes_loco import PANICLE_BayesLOCO


@dataclass(frozen=True)
class BenchmarkSize:
    n_samples: int
    n_markers: int


@dataclass
class SyntheticDataset:
    phe: np.ndarray
    geno: np.ndarray
    map_df: pd.DataFrame
    covariates: np.ndarray
    causal_indices: np.ndarray


def _parse_sizes(text: str) -> List[BenchmarkSize]:
    out: List[BenchmarkSize] = []
    for token in (x.strip() for x in text.split(",")):
        if not token:
            continue
        parts = token.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid size token '{token}', expected NxM")
        n = int(parts[0])
        m = int(parts[1])
        if n <= 0 or m <= 0:
            raise ValueError(f"Invalid size token '{token}', N and M must be > 0")
        out.append(BenchmarkSize(n_samples=n, n_markers=m))
    if not out:
        raise ValueError("No valid sizes parsed from --sizes")
    return out


def _parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_float_tuple(text: str) -> Tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise ValueError("Expected at least one numeric value")
    return vals


def _build_map(n_markers: int, n_chromosomes: int) -> pd.DataFrame:
    blocks = np.repeat(np.arange(1, n_chromosomes + 1), int(np.ceil(n_markers / n_chromosomes)))[:n_markers]
    return pd.DataFrame(
        {
            "SNP": [f"SNP{i:07d}" for i in range(n_markers)],
            "CHROM": [f"Chr{int(c):02d}" for c in blocks],
            "POS": np.arange(1, n_markers + 1, dtype=np.int64),
        }
    )


def _build_dataset(
    *,
    seed: int,
    size: BenchmarkSize,
    scenario: str,
    n_covariates: int,
    n_chromosomes: int,
    n_causal: int,
    effect_scale: float,
    missing_rate: float,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    n = size.n_samples
    m = size.n_markers

    geno = rng.integers(0, 3, size=(n, m), dtype=np.int8)
    if missing_rate > 0.0:
        miss = rng.random((n, m)) < missing_rate
        geno = geno.copy()
        geno[miss] = -9

    covariates = rng.normal(size=(n, n_covariates)).astype(np.float64)
    cov_beta = rng.normal(scale=0.3, size=n_covariates)
    y = covariates @ cov_beta + rng.normal(scale=1.0, size=n)

    causal_indices = np.array([], dtype=np.int64)
    if scenario == "power":
        k = min(max(1, n_causal), m)
        causal_indices = np.sort(rng.choice(m, size=k, replace=False).astype(np.int64))
        signs = rng.choice(np.array([-1.0, 1.0]), size=k)
        effects = signs * np.abs(rng.normal(loc=effect_scale, scale=max(effect_scale * 0.2, 1e-6), size=k))
        y = y + geno[:, causal_indices].astype(np.float64) @ effects

    phe = np.column_stack([np.array([f"id{i:07d}" for i in range(n)]), y])
    map_df = _build_map(m, n_chromosomes)
    return SyntheticDataset(
        phe=phe,
        geno=geno,
        map_df=map_df,
        covariates=covariates,
        causal_indices=causal_indices,
    )


def _safe_float(x: Any) -> float:
    try:
        val = float(x)
    except Exception:
        return float("nan")
    if np.isfinite(val):
        return val
    return float("nan")


def _scenario_metrics(pvalues: np.ndarray, causal_indices: np.ndarray, scenario: str) -> Dict[str, float]:
    p = np.asarray(pvalues, dtype=np.float64)
    p = np.clip(p, 1e-300, 1.0)

    out: Dict[str, float] = {
        "min_p": float(np.min(p)),
        "median_p": float(np.median(p)),
        "frac_p_lt_0p05": float(np.mean(p < 0.05)),
        "frac_p_lt_1e3": float(np.mean(p < 1e-3)),
    }

    if scenario != "power" or causal_indices.size == 0:
        out["causal_best_rank"] = float("nan")
        out["causal_frac_p_lt_1e4"] = float("nan")
        out["causal_mean_neglog10p"] = float("nan")
        out["null_mean_neglog10p"] = float("nan")
        return out

    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, p.size + 1)

    causal_p = p[causal_indices]
    non_causal_mask = np.ones(p.size, dtype=bool)
    non_causal_mask[causal_indices] = False
    non_causal_p = p[non_causal_mask]

    out["causal_best_rank"] = float(np.min(ranks[causal_indices]))
    out["causal_frac_p_lt_1e4"] = float(np.mean(causal_p < 1e-4))
    out["causal_mean_neglog10p"] = float(np.mean(-np.log10(causal_p)))
    out["null_mean_neglog10p"] = float(np.mean(-np.log10(non_causal_p))) if non_causal_p.size > 0 else float("nan")
    return out


def _build_seed(
    *,
    base_seed: int,
    size: BenchmarkSize,
    replicate: int,
    scenario: str,
    loco_mode: str,
    calibration_mode: str,
) -> int:
    scenario_off = 17 if scenario == "power" else 3
    loco_off = 29 if loco_mode == "refine" else 11
    calib_off = {"none": 5, "gc": 13, "unrelated_subset": 23}[calibration_mode]
    return int(
        base_seed
        + replicate * 1_000_003
        + size.n_samples * 101
        + size.n_markers * 17
        + scenario_off
        + loco_off
        + calib_off
    )


def _run_one(
    *,
    size: BenchmarkSize,
    replicate: int,
    scenario: str,
    loco_mode: str,
    calibration_mode: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    seed = _build_seed(
        base_seed=args.base_seed,
        size=size,
        replicate=replicate,
        scenario=scenario,
        loco_mode=loco_mode,
        calibration_mode=calibration_mode,
    )

    ds = _build_dataset(
        seed=seed,
        size=size,
        scenario=scenario,
        n_covariates=args.n_covariates,
        n_chromosomes=args.n_chromosomes,
        n_causal=args.n_causal,
        effect_scale=args.effect_scale,
        missing_rate=args.missing_rate,
    )

    bl_cfg: Dict[str, Any] = {
        "max_iter": args.max_iter,
        "patience": args.patience,
        "prior_tune_stage1_max_iter": args.prior_tune_stage1_max_iter,
        "prior_tune_stage2_max_iter": args.prior_tune_stage2_max_iter,
        "prior_tune_pi_grid": args.prior_pi_grid,
        "prior_tune_slab_scale_grid": args.prior_slab_scale_grid,
        "prior_tune_top_k": args.prior_tune_top_k,
        "batch_markers_fit": args.batch_markers_fit,
        "batch_markers_test": args.batch_markers_test,
        "screening_warmup_epochs": args.screening_warmup_epochs,
        "verification_interval": args.verification_interval,
        "loco_mode": loco_mode,
        "loco_refine_iter": args.loco_refine_iter,
        "refine_patience": args.refine_patience,
        "test_method": args.test_method,
        "robust_se": bool(args.robust_se),
        "calibrate_stat_scale": calibration_mode,
        "random_seed": seed,
        "deterministic": True,
    }

    if calibration_mode == "unrelated_subset":
        subset_n = int(round(size.n_samples * args.unrelated_fraction))
        subset_n = max(subset_n, args.unrelated_min_n)
        subset_n = min(subset_n, size.n_samples)
        subset_n = max(10, subset_n)
        unrelated_idx = list(np.arange(subset_n, dtype=np.int64))
        bl_cfg["unrelated_subset_indices"] = unrelated_idx
        bl_cfg["unrelated_subset_min_n"] = min(subset_n, max(10, args.unrelated_min_n))

    t0 = time.perf_counter()
    res = PANICLE_BayesLOCO(
        phe=ds.phe,
        geno=ds.geno,
        map_data=ds.map_df,
        CV=ds.covariates,
        cpu=1,
        verbose=False,
        bl_config=bl_cfg,
    )
    wall_s = time.perf_counter() - t0

    md = dict(res.metadata or {})
    row: Dict[str, Any] = {
        "n_samples": size.n_samples,
        "n_markers": size.n_markers,
        "replicate": replicate,
        "scenario": scenario,
        "loco_mode": loco_mode,
        "calibration_mode_requested": calibration_mode,
        "calibration_mode_effective": md.get("calibration_mode", calibration_mode),
        "seed": seed,
        "wall_s": float(wall_s),
        "timing_total_s": _safe_float(md.get("timing_total_s")),
        "timing_prior_tune_s": _safe_float(md.get("timing_prior_tune_s")),
        "timing_main_fit_s": _safe_float(md.get("timing_main_fit_s")),
        "timing_loco_test_s": _safe_float(md.get("timing_loco_test_s")),
        "pass_equiv_total": _safe_float(md.get("pass_equiv_total")),
        "pass_equiv_prior_tune": _safe_float(md.get("pass_equiv_prior_tune")),
        "pass_equiv_main_fit": _safe_float(md.get("pass_equiv_main_fit")),
        "pass_equiv_loco_refine": _safe_float(md.get("pass_equiv_loco_refine")),
        "lambda_gc_raw": _safe_float(md.get("lambda_gc_raw")),
        "lambda_gc_final": _safe_float(md.get("lambda_gc_final")),
        "h2_hat": _safe_float(md.get("h2_hat")),
        "prior_pi_selected": _safe_float(md.get("prior_pi_selected")),
        "prior_slab_scale_selected": _safe_float(md.get("prior_slab_scale_selected")),
        "sigma_slab2_selected": _safe_float(md.get("sigma_slab2_selected")),
        "sigma_e2_final": _safe_float(md.get("sigma_e2_final")),
        "converged": bool(md.get("converged", False)),
        "n_markers_fit": int(md.get("n_markers_fit", size.n_markers)),
        "unrelated_subset_n": int(md.get("unrelated_subset_n", 0)),
    }
    row.update(_scenario_metrics(res.pvalues, ds.causal_indices, scenario))
    return row


def _aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = ["n_samples", "n_markers", "scenario", "loco_mode", "calibration_mode_requested"]
    metric_cols = [
        "wall_s",
        "timing_total_s",
        "timing_prior_tune_s",
        "timing_main_fit_s",
        "timing_loco_test_s",
        "pass_equiv_total",
        "lambda_gc_raw",
        "lambda_gc_final",
        "frac_p_lt_0p05",
        "frac_p_lt_1e3",
        "causal_best_rank",
        "causal_frac_p_lt_1e4",
        "causal_mean_neglog10p",
        "null_mean_neglog10p",
    ]
    agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max", "count"])
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    return agg.to_dict(orient="records")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _print_compact_summary(summary_rows: Sequence[Dict[str, Any]]) -> None:
    if not summary_rows:
        print("No benchmark rows produced.")
        return
    print("\nSummary (mean wall_s / mean lambda_gc_final / mean frac_p_lt_0p05):")
    for row in summary_rows:
        print(
            f"  n={row['n_samples']} m={row['n_markers']} "
            f"{row['scenario']} {row['loco_mode']} {row['calibration_mode_requested']}: "
            f"wall={row.get('wall_s_mean', float('nan')):.3f}s, "
            f"lambda={row.get('lambda_gc_final_mean', float('nan')):.3f}, "
            f"p<0.05={row.get('frac_p_lt_0p05_mean', float('nan')):.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BAYESLOCO on synthetic datasets")
    parser.add_argument(
        "--sizes",
        type=str,
        default="120x2000,240x5000",
        help="Comma-separated NxM list (examples: 120x2000,240x5000)",
    )
    parser.add_argument("--replicates", type=int, default=2, help="Replicates per size/scenario/mode")
    parser.add_argument("--base-seed", type=int, default=20260217, help="Base seed")

    parser.add_argument("--scenarios", type=str, default="null,power", help="Comma-separated scenarios: null,power")
    parser.add_argument("--loco-modes", type=str, default="subtract_only,refine", help="Comma-separated LOCO modes")
    parser.add_argument(
        "--calibration-modes",
        type=str,
        default="gc",
        help="Comma-separated calibration modes: none,gc,unrelated_subset",
    )

    parser.add_argument("--n-covariates", type=int, default=3)
    parser.add_argument("--n-chromosomes", type=int, default=12)
    parser.add_argument("--n-causal", type=int, default=8, help="Number of causal markers for power scenario")
    parser.add_argument("--effect-scale", type=float, default=0.6, help="Approximate effect scale for causal markers")
    parser.add_argument("--missing-rate", type=float, default=0.01, help="Synthetic genotype missing rate")

    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--prior-tune-stage1-max-iter", type=int, default=6)
    parser.add_argument("--prior-tune-stage2-max-iter", type=int, default=8)
    parser.add_argument("--prior-tune-top-k", type=int, default=3)
    parser.add_argument("--prior-pi-grid", type=str, default="0.005,0.02,0.05")
    parser.add_argument("--prior-slab-scale-grid", type=str, default="0.75,1.25")
    parser.add_argument("--batch-markers-fit", type=int, default=2048)
    parser.add_argument("--batch-markers-test", type=int, default=8192)
    parser.add_argument("--screening-warmup-epochs", type=int, default=6)
    parser.add_argument("--verification-interval", type=int, default=8)
    parser.add_argument("--loco-refine-iter", type=int, default=8)
    parser.add_argument("--refine-patience", type=int, default=3)
    parser.add_argument("--test-method", type=str, choices=["score", "wald"], default="score")
    parser.add_argument("--robust-se", action="store_true", help="Enable robust SE on Wald path")

    parser.add_argument("--unrelated-fraction", type=float, default=0.5, help="Subset fraction for unrelated calibration")
    parser.add_argument("--unrelated-min-n", type=int, default=50, help="Minimum unrelated subset size")

    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sizes = _parse_sizes(args.sizes)
    scenarios = _parse_str_list(args.scenarios)
    loco_modes = _parse_str_list(args.loco_modes)
    calibration_modes = _parse_str_list(args.calibration_modes)

    valid_scenarios = {"null", "power"}
    valid_loco = {"subtract_only", "refine"}
    valid_calibration = {"none", "gc", "unrelated_subset"}
    if not set(scenarios).issubset(valid_scenarios):
        raise ValueError(f"--scenarios must be subset of {sorted(valid_scenarios)}")
    if not set(loco_modes).issubset(valid_loco):
        raise ValueError(f"--loco-modes must be subset of {sorted(valid_loco)}")
    if not set(calibration_modes).issubset(valid_calibration):
        raise ValueError(f"--calibration-modes must be subset of {sorted(valid_calibration)}")

    args.prior_pi_grid = _parse_float_tuple(args.prior_pi_grid)
    args.prior_slab_scale_grid = _parse_float_tuple(args.prior_slab_scale_grid)

    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")
    if not (0.0 < args.unrelated_fraction <= 1.0):
        raise ValueError("--unrelated-fraction must be in (0, 1]")

    rows: List[Dict[str, Any]] = []
    run_total = len(sizes) * args.replicates * len(scenarios) * len(loco_modes) * len(calibration_modes)
    run_idx = 0

    for size in sizes:
        for rep in range(args.replicates):
            for scenario in scenarios:
                for loco_mode in loco_modes:
                    for calibration_mode in calibration_modes:
                        run_idx += 1
                        if not args.quiet:
                            print(
                                f"[{run_idx}/{run_total}] n={size.n_samples} m={size.n_markers} "
                                f"rep={rep} scenario={scenario} loco={loco_mode} calib={calibration_mode}"
                            )
                        row = _run_one(
                            size=size,
                            replicate=rep,
                            scenario=scenario,
                            loco_mode=loco_mode,
                            calibration_mode=calibration_mode,
                            args=args,
                        )
                        rows.append(row)

    summary_rows = _aggregate(rows)
    _print_compact_summary(summary_rows)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "sizes": [asdict(s) for s in sizes],
            "replicates": args.replicates,
            "scenarios": scenarios,
            "loco_modes": loco_modes,
            "calibration_modes": calibration_modes,
            "n_covariates": args.n_covariates,
            "n_chromosomes": args.n_chromosomes,
            "n_causal": args.n_causal,
            "effect_scale": args.effect_scale,
            "missing_rate": args.missing_rate,
            "max_iter": args.max_iter,
            "patience": args.patience,
            "prior_tune_stage1_max_iter": args.prior_tune_stage1_max_iter,
            "prior_tune_stage2_max_iter": args.prior_tune_stage2_max_iter,
            "prior_tune_top_k": args.prior_tune_top_k,
            "prior_pi_grid": args.prior_pi_grid,
            "prior_slab_scale_grid": args.prior_slab_scale_grid,
            "batch_markers_fit": args.batch_markers_fit,
            "batch_markers_test": args.batch_markers_test,
            "screening_warmup_epochs": args.screening_warmup_epochs,
            "verification_interval": args.verification_interval,
            "loco_refine_iter": args.loco_refine_iter,
            "refine_patience": args.refine_patience,
            "test_method": args.test_method,
            "robust_se": bool(args.robust_se),
            "base_seed": args.base_seed,
        },
        "rows": rows,
        "summary": summary_rows,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(_to_serializable(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON report: {args.output_json}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"Wrote CSV rows: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

