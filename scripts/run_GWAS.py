#!/usr/bin/env python3
"""Compatibility wrapper for the packaged ``panicle-gwas`` CLI.

Prefer::

    panicle-gwas --phenotype ... --genotype ...
    python -m panicle --phenotype ... --genotype ...

This script remains for existing workflows and documentation that cite
``scripts/run_GWAS.py``.
"""

from panicle.cli.gwas import main

if __name__ == "__main__":
    raise SystemExit(main())
