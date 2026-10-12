# Zinnia

Zinnia compiles Python programs that use NumPy into zero-knowledge circuits. A function
decorated with `@zk_circuit` is executed symbolically, turned into an IR, optimized, and
lowered to one of two proving backends: Halo2-lib (`halo2-base`) or Noir. The same function
can also be run on a mock backend that evaluates the circuit without proving.

```python
from zinnia import zk_circuit, NDArray, ZKCircuit

@zk_circuit
def dot(x: NDArray[int, 4], y: NDArray[int, 4], expected: int):
    assert (x * y).sum() == expected

assert dot([1, 2, 3, 4], [5, 6, 7, 8], 70)     # calling the function runs it on the mock backend

circuit = ZKCircuit.from_method(dot)
proof = circuit.prove([1, 2, 3, 4], [5, 6, 7, 8], 70, backend="halo2-base")   # or "noir"
assert circuit.verify(proof)
```

## Repository layout

| Path | Contents |
|---|---|
| `zinnia/` | Python package: user API (`zk_circuit`, `zk_chip`, `zk_external`, `requires`), the front end that turns the decorated source into the JSON AST the core reads, and the input parser |
| `src/` | Rust core (the `_zinnia_core` extension): symbolic execution and IR generation (`ir_gen/`, `builder/`), IR (`ir*.rs`), optimization passes (`optim/`), array lowering (`lower/`), backends (`prove/`: mock, `halo2_base`, `noir`) |
| `testing/` | Python test suite (pytest) |
| `tests/` | Rust integration test of the Noir backend |
| `examples/` | Small example circuits |
| `docs-site/` | User and contributor documentation (Fumadocs site) |
| `docs/` | Design notes on array facts, array versions, ablation switches and helpers |
| `benchmarking/` | Benchmark programs, baselines and the benchmark harness (see `benchmarking/README.md`) |
| `eval/` | Data and scripts behind the paper's evaluation (see `eval/README.md`) |
| `studies/` | User-study materials |

## Building

Zinnia needs Python 3.10 or newer, a Rust toolchain, and `maturin`. The Rust core links
`libz3`; `scripts/develop.sh` finds the copy shipped in the `z3-solver` wheel.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt "maturin>=1,<2"

# Mock backend only (debug build)
PYTHON=.venv/bin/python scripts/develop.sh

# With the halo2-base and Noir proving backends (release build)
RUSTC_BOOTSTRAP=1 PYTHON=.venv/bin/python scripts/develop.sh --release \
    --features halo2-base-backend,noir-backend
```

`RUSTC_BOOTSTRAP=1` is needed because a dependency of `halo2-base` uses unstable Rust
features. The Noir backend runs `nargo` and `bb` (Barretenberg), found through the
`ZINNIA_NARGO_BIN` and `ZINNIA_BB_BIN` environment variables, then in `~/.nargo/bin` and `~/.bb`,
then on `PATH`; the benchmark harness pins nargo `1.0.0-beta.20`
(`benchmarking/harness/toolchains.py`).

## Testing

```bash
.venv/bin/python -m pytest                     # Python tests under testing/
scripts/check_tests_compile.sh                 # type-check the Rust tests
```

To run the Rust library tests, point the linker at the wheel's `libz3` (the directory that
contains `libz3.so` inside the `z3` package of the venv) and skip the slow proving tests:

```bash
Z3=$(.venv/bin/python -c "import z3, pathlib; print(next(pathlib.Path(z3.__file__).parent.glob('**/libz3.so')).parent)")
LIBRARY_PATH=$Z3 LD_LIBRARY_PATH=$Z3 PYO3_PYTHON=.venv/bin/python RUSTC_BOOTSTRAP=1 \
    cargo test --release --lib --features halo2-base-backend,noir-backend -- --skip prove::
```

## Documentation

The documentation site lives in `docs-site/` (`npm install && npm run dev` inside it). It covers
the language (types, operators, arrays, control flow, privacy annotations), the programming
guide (circuits, chips, external functions, execution, configuration, NumPy API) and the
compiler internals.
