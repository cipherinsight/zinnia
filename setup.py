"""Legacy setuptools entry point.

Zinnia is a Rust-backed Python extension built by maturin through
pyproject.toml. A plain setuptools install would miss the native module, so
fail early with the supported commands instead of creating a broken package.
"""

raise SystemExit(
    "setup.py is intentionally disabled. Use `scripts/develop.sh` for a local "
    "editable build, or use `python -m pip install .` / `python -m pip install "
    "-e .` so pip invokes the maturin backend from pyproject.toml."
)
