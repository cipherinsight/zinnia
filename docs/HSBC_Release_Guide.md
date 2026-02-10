# Zinnia HSBC Release Guide

## Release Date
2026-02-10

## Package
- Name: `zinnia`
- Version: `0.0.1`
- Wheel: `dist/zinnia-0.0.1-py3-none-any.whl`

## 1. Environment Setup
1. Install Python 3.10+.
2. Create and activate a virtual environment.
3. Install the wheel:
   - `python -m pip install zinnia-0.0.1-py3-none-any.whl`

Optional dependency set for validation:
- `python -m pip install -r requirements.txt`
- `python -m pip install scipy`

## 2. Quick Verification
Run:
- `python -c "import zinnia; print('zinnia import OK')"`

Expected output:
- `zinnia import OK`

## 3. Usage Example
```python
from zinnia.api import zk_circuit

@zk_circuit
def add_and_check(x: int, y: int, result: int):
    assert x + y == result

ok = add_and_check(2, 3, 5)
print(bool(ok))
```

## 4. Project Commands
- Run tests: `pytest`
- Type checks and formatting checks: `bash type-check.sh`

## 5. Test Status for This Release
Executed on 2026-02-10 with Python 3.10 and `pytest 8.3.4`.
- Total: 279
- Passed: 274
- Skipped: 2
- Failed: 3

Failing tests:
- `testing/operator/lst/test_list_index.py::test_list_index`
- `testing/operator/lst/test_list_remove.py::test_list_remove_not_exists_dynamic`
- `testing/operator/tupl/test_tuple_index.py::test_tuple_index`

## 6. Release Contents
- Wheel package (`dist/`)
- HSBC release guide (this document, plus PDF copy)
- Source code and tests

