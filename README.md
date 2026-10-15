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

Zinnia needs Python 3.10 or newer, a stable Rust toolchain and `maturin` (installed below).
The Rust core links `libz3`; `build.rs` finds the copy shipped in the `z3-solver` wheel of
the Python interpreter it builds for, so no system Z3 is needed. `halo2-base` and its
dependency `poseidon-primitives` use unstable Rust features, which `.cargo/config.toml`
enables for those two crates alone (through `RUSTC_BOOTSTRAP`), so the proving backends
also build on a stable toolchain.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

scripts/develop.sh                                                  # mock backend only (debug build)
scripts/develop.sh --release --features halo2-base-backend,noir-backend   # with the proving backends
```

`scripts/develop.sh` runs `maturin develop` for the active interpreter and passes its
arguments through. The Noir backend runs `nargo` and `bb` (Barretenberg), found through the
`ZINNIA_NARGO_BIN` and `ZINNIA_BB_BIN` environment variables, then in `~/.nargo/bin` and `~/.bb`,
then on `PATH`; the benchmark harness pins nargo `1.0.0-beta.20`
(`benchmarking/harness/toolchains.py`).

## Testing

With the virtual environment active and the extension built:

```bash
pytest                                     # Python tests under testing/, on every core
cargo test --release --lib --features halo2-base-backend,noir-backend -- --skip prove::
scripts/check_tests_compile.sh             # type-check the Rust tests only
scripts/check_build_matrix.sh              # type-check every backend feature combination
```

The Rust library tests under `prove::` generate keys and proofs and are slow; drop the
`--skip` to run them too.

## Documentation

The documentation site lives in `docs-site/` (`npm install && npm run dev` inside it). It covers
the language (types, operators, arrays, control flow, privacy annotations), the programming
guide (circuits, chips, external functions, execution, configuration, NumPy API) and the
compiler internals.
