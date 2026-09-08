# Zinnia

Zinnia is a Python-based framework for building simple and efficient zero-knowledge proof circuits.

## Installation

- Create a virtual environment and install the Python dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install "maturin>=1,<2"
```

- Build the native extension in-place:
```bash
scripts/develop.sh
```

`scripts/develop.sh` also handles local libz3 discovery from the `z3-solver`
wheel and the PyO3 forward-compatibility flag needed by Python 3.14.

- Check the installation:
```bash
python -m pytest
```

## Documentation

You may find the documentation for Zinnia in the `docs-site` folder.

## Examples

You may find the examples for Zinnia in the `examples` folder.
