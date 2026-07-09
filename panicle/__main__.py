"""Allow ``python -m panicle`` to invoke the GWAS CLI."""

from panicle.cli.gwas import main

if __name__ == "__main__":
    raise SystemExit(main())
