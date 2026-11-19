"""Allow running Exo CLI via `python3 -m exo`."""

from exo.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
