"""CLI utility to cache genotype files into fast-loading memmap format."""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Sequence

import numpy as np

from ..data.loaders import load_genotype_file, detect_file_format
from ..utils.memmap_utils import save_genotype_to_memmap


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    return raw


def _parse_loader_options(options: Sequence[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in options:
        if '=' not in item:
            raise ValueError(f"Loader option '{item}' must be in KEY=VALUE format")
        key, value = item.split('=', 1)
        parsed[key] = _coerce_value(value)
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert genotype files (VCF, HapMap, CSV, etc.) into a cached memmap for fast reuse.",
    )
    parser.add_argument('-i', '--input', required=True, help='Path to the input genotype file')
    parser.add_argument('-o', '--output-prefix', required=True, help='Output prefix for cached artefacts')
    parser.add_argument('--format', choices=['vcf', 'plink', 'hapmap', 'csv', 'tsv', 'numeric', 'memmap'],
                        help='Explicitly specify input file format (default: auto-detect)')
    parser.add_argument('--dtype', default='int8', help='Storage dtype for the cached memmap (default: int8)')
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='Number of markers to copy per batch when writing the cache')
    parser.add_argument('--load-option', action='append', default=[],
                        help='Additional loader option of the form KEY=VALUE (repeatable)')
    parser.add_argument('--precompute-alleles', dest='precompute_alleles', action='store_true',
                        help='Precompute major alleles while loading (default: off for conversion)')
    parser.add_argument('--no-precompute-alleles', dest='precompute_alleles', action='store_false',
                        help='Disable major-allele precomputation during load')
    parser.set_defaults(precompute_alleles=False)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file '{input_path}' does not exist")

    output_prefix = Path(args.output_prefix)
    loader_kwargs = _parse_loader_options(args.load_option)
    loader_kwargs.setdefault('precompute_alleles', args.precompute_alleles)

    file_format = args.format or detect_file_format(input_path)
    if file_format == 'unknown':
        parser.error("Unable to determine input file format; please specify via --format")

    try:
        dtype = np.dtype(args.dtype)
    except TypeError as exc:
        parser.error(f"Unsupported dtype '{args.dtype}': {exc}")

    genotype, sample_ids, geno_map = load_genotype_file(
        input_path,
        file_format=file_format,
        **loader_kwargs,
    )

    result = save_genotype_to_memmap(
        genotype=genotype,
        sample_ids=sample_ids,
        geno_map=geno_map,
        output_prefix=output_prefix,
        dtype=dtype,
        batch_size=args.batch_size,
    )

    n_individuals, n_markers = result['shape']
    print(f"Cached genotype matrix: {n_individuals} individuals × {n_markers} markers")
    print(f"  Memmap: {result['memmap_path']}")
    print(f"  Metadata: {result['metadata_path']}")
    print(f"  Samples: {result['samples_path']}")
    if result['map_path'] is not None:
        print(f"  Map: {result['map_path']}")
    else:
        print("  Map: not provided")
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
