"""PANICLE GWAS command-line interface.

Installed as ``panicle-gwas`` and also available via::

    python -m panicle
    python scripts/run_GWAS.py   # thin compatibility wrapper
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

from panicle.cli.utils import parse_args
from panicle.pipelines.gwas import GWASPipeline

OUTPUT_CHOICES = (
    "all_marker_pvalues",
    "significant_marker_pvalues",
    "manhattan",
    "qq",
)


def normalize_outputs(outputs):
    """Normalize output selections with comma splitting and deduplication."""
    if not outputs:
        return list(OUTPUT_CHOICES)

    normalized = []
    seen = set()
    for item in outputs:
        for part in str(item).split(","):
            part = part.strip().lower()
            if not part:
                continue
            if part not in OUTPUT_CHOICES:
                raise ValueError(f"Invalid output choice: {part}")
            if part not in seen:
                normalized.append(part)
                seen.add(part)

    return normalized if normalized else list(OUTPUT_CHOICES)


def normalize_methods(methods):
    """Normalize method names to pipeline-supported identifiers."""
    if not methods:
        return []

    aliases = {
        "GLM": "GLM",
        "MLM": "MLM",
        "BAYESLOCO": "BAYESLOCO",
        "FARMCPU": "FARMCPU",
        "FARMCPU_RESAMPLING": "FarmCPUResampling",
        "FARMCPURESAMPLING": "FarmCPUResampling",
        "RESAMPLING": "FarmCPUResampling",
        "BLINK": "BLINK",
    }

    normalized = []
    seen = set()
    for m in methods:
        key = str(m).replace("-", "_").replace(" ", "_").strip().upper()
        if not key:
            continue
        method = aliases.get(key, key)
        if method not in seen:
            normalized.append(method)
            seen.add(method)
    return normalized


def _parse_float_tuple(text: Optional[str]) -> Optional[Tuple[float, ...]]:
    if text is None:
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        return None
    return tuple(float(p) for p in parts)


def _load_index_file(path: Optional[str]) -> Optional[list]:
    if path is None:
        return None
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            out.append(int(token))
    return out


def _requires_global_kinship(valid_methods, pipeline, mlm_mode: str = "loco") -> bool:
    """Return whether the selected methods need a precomputed global kinship."""
    if "MLM" not in valid_methods:
        return False
    mode = str(mlm_mode or "loco").strip().lower()
    if mode == "global":
        return True
    # LOCO needs a map; without one, fall back to global kinship.
    return getattr(pipeline, "geno_map", None) is None


class FarmCPUResamplingProgressReporter:
    """Lightweight progress reporter for FarmCPU resampling runs."""

    def __init__(self, trait_name: str, total_runs: int):
        self.trait_name = trait_name
        self.total_runs = total_runs
        self._started = False

    def __call__(self, run_idx: int, total_runs: int, elapsed_seconds: float) -> None:
        if not self._started:
            print(f"[{self.trait_name}] started resampling ({total_runs} runs)")
            self._started = True

        if run_idx >= total_runs:
            print(f"[{self.trait_name}] finished resampling in {elapsed_seconds:.0f}s")
            return

        threshold = max(5, int(0.1 * total_runs))
        if run_idx >= threshold:
            remaining = max(total_runs - run_idx, 0)
            avg_per_run = elapsed_seconds / max(run_idx, 1)
            eta = remaining * avg_per_run
            print(
                f"[{self.trait_name}] progress {run_idx}/{total_runs} "
                f"(ETA {eta:.0f}s)"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the PANICLE GWAS CLI.

    Parameters
    ----------
    argv :
        Argument list without the program name (same convention as
        ``argparse.ArgumentParser.parse_args``).  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    args = parse_args(argv)

    # Backward compatibility: older CLI definitions may not include method flags.
    # Default them to False so downstream logic can fall back to --methods.
    for flag in ("glm", "mlm", "farmcpu", "resampling"):
        if not hasattr(args, flag):
            setattr(args, flag, False)

    pipeline = GWASPipeline(output_dir=args.outputdir)

    traits = [t.strip() for t in args.traits.split(",")] if args.traits else None
    cov_cols = (
        [c.strip() for c in args.covariate_columns.split(",")]
        if args.covariate_columns
        else None
    )

    loader_kwargs = {
        "drop_monomorphic": args.drop_monomorphic,
        "max_missing": args.max_missing,
        "min_maf": args.min_maf,
        "include_indels": not args.snps_only,
        "split_multiallelic": not args.no_split_multiallelic,
        "compute_effective_tests": args.compute_effective_tests,
    }
    if args.compute_effective_tests:
        if args.parallel_mode == "off":
            effective_ncpus = 1
        elif args.ncpus == 0:
            effective_ncpus = max(1, os.cpu_count() or 1)
        else:
            effective_ncpus = max(1, int(args.ncpus))
        loader_kwargs["effective_test_kwargs"] = {"ncpus": effective_ncpus}

    pipeline.load_data(
        phenotype_file=args.phenotype,
        phenotype_id_column=args.phenotype_id_column,
        genotype_file=args.genotype,
        map_file=args.map,
        genotype_format=args.format,
        trait_columns=traits,
        covariate_file=args.covariates,
        covariate_columns=cov_cols,
        covariate_id_column=args.covariate_id_column,
        loader_kwargs=loader_kwargs,
    )

    pipeline.align_samples()

    methods: List[str] = []
    if args.glm:
        methods.append("GLM")
    if args.mlm:
        methods.append("MLM")
    if args.farmcpu:
        methods.append("FARMCPU")
    if args.resampling:
        methods.append("FarmCPUResampling")

    if args.methods:
        methods.extend([m for m in args.methods.split(",") if m.strip()])

    valid_methods = normalize_methods(methods)
    outputs = normalize_outputs(args.outputs)

    need_kinship = _requires_global_kinship(
        valid_methods, pipeline, mlm_mode=args.mlm_mode
    )
    pipeline.compute_population_structure(
        n_pcs=args.n_pcs, calculate_kinship=need_kinship
    )

    def _resolve_denom() -> float:
        if args.n_eff:
            return float(args.n_eff)
        if args.compute_effective_tests:
            if pipeline.effective_tests_info and pipeline.effective_tests_info.get("Me"):
                return float(pipeline.effective_tests_info["Me"])
        return float(pipeline.genotype_matrix.n_markers)

    farmcpu_params = {
        "resampling_runs": args.farmcpu_resampling_runs,
        "resampling_mask_proportion": args.farmcpu_resampling_mask_proportion,
        "resampling_cluster_markers": args.farmcpu_resampling_cluster,
        "resampling_ld_threshold": args.farmcpu_resampling_ld_threshold,
    }
    if args.farmcpu_resampling_seed is not None:
        farmcpu_params["resampling_random_seed"] = args.farmcpu_resampling_seed

    resampling_threshold = args.farmcpu_resampling_significance
    if resampling_threshold is None and args.farmcpu_resampling_alpha is not None:
        denom = _resolve_denom()
        resampling_threshold = args.farmcpu_resampling_alpha / max(denom, 1.0)

    if resampling_threshold is not None:
        farmcpu_params["resampling_significance_threshold"] = resampling_threshold

    qtn_threshold = args.farmcpu_qtn_threshold
    qtn_threshold_is_corrected = False
    if qtn_threshold is None and args.farmcpu_qtn_alpha is not None:
        denom = _resolve_denom()
        qtn_threshold = args.farmcpu_qtn_alpha / max(denom, 1.0)
        qtn_threshold_is_corrected = True
    if qtn_threshold is not None:
        farmcpu_params["QTN_threshold"] = qtn_threshold
        if args.farmcpu_qtn_threshold is not None:
            qtn_threshold_is_corrected = True
        if qtn_threshold_is_corrected:
            farmcpu_params["QTN_threshold_is_corrected"] = True
        if args.farmcpu_p_threshold is None:
            farmcpu_params["p_threshold"] = qtn_threshold

    if args.farmcpu_p_threshold is not None:
        farmcpu_params["p_threshold"] = args.farmcpu_p_threshold

    if "FarmCPUResampling" in valid_methods:
        farmcpu_params.setdefault("resampling_progress", True)

    bayesloco_params = {}
    if args.bayesloco_max_iter is not None:
        bayesloco_params["max_iter"] = int(args.bayesloco_max_iter)
    if args.bayesloco_patience is not None:
        bayesloco_params["patience"] = int(args.bayesloco_patience)
    if args.bayesloco_loco_mode is not None:
        bayesloco_params["loco_mode"] = args.bayesloco_loco_mode
    if args.bayesloco_test_method is not None:
        bayesloco_params["test_method"] = args.bayesloco_test_method
    if args.bayesloco_calibration is not None:
        bayesloco_params["calibrate_stat_scale"] = args.bayesloco_calibration
    if args.bayesloco_batch_fit is not None:
        bayesloco_params["batch_markers_fit"] = int(args.bayesloco_batch_fit)
    if args.bayesloco_batch_test is not None:
        bayesloco_params["batch_markers_test"] = int(args.bayesloco_batch_test)
    if args.bayesloco_refine_iter is not None:
        bayesloco_params["loco_refine_iter"] = int(args.bayesloco_refine_iter)

    pi_grid = _parse_float_tuple(args.bayesloco_prior_pi_grid)
    if pi_grid is not None:
        bayesloco_params["prior_tune_pi_grid"] = pi_grid
    slab_grid = _parse_float_tuple(args.bayesloco_prior_slab_scale_grid)
    if slab_grid is not None:
        bayesloco_params["prior_tune_slab_scale_grid"] = slab_grid

    unrelated_indices = _load_index_file(args.bayesloco_unrelated_indices)
    if unrelated_indices is not None:
        bayesloco_params["unrelated_subset_indices"] = unrelated_indices

    pipeline.run_analysis(
        traits=traits,
        methods=valid_methods,
        max_iterations=args.max_iterations,
        ncpus=args.ncpus,
        parallel_mode=args.parallel_mode,
        significance=args.significance,
        alpha=args.alpha,
        n_eff=args.n_eff,
        max_genotype_dosage=args.max_genotype_dosage,
        min_mac=args.min_mac,
        mlm_mode=args.mlm_mode,
        farmcpu_params=farmcpu_params,
        bayesloco_params=(bayesloco_params or None),
        include_standard_errors=args.include_standard_errors,
        outputs=outputs,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
