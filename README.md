# Zinnia

Zinnia is a Python framework for building Zero-Knowledge (ZK) circuits for privacy-preserving data-analytics workloads.

This repository is being released as an artifact for **HSBC** review.

---

## 1) HSBC Review Scope

For this artifact review, the expected workflow is:

1. Review the example circuits in `examples/`.
2. Execute the examples end-to-end (compile → keygen → prove → verify).
3. Inspect generated outputs / logs.

> The execution environment is provisioned in Docker. You do **not** need to install dependencies manually from this repository for normal evaluation.

---

## 2) Quick Start

From the repository root, run any example directly:

```bash
python examples/proof_of_solvency.py
python examples/aml_structuring_detection_proof.py
python examples/private_credit_policy_check.py
python examples/private_kpi_benchmarking.py
python examples/sanctions_pep_screening.py
python examples/membership_proof.py
```

Each script performs the canonical flow:

1. Build circuit object (`create_circuit(...)`)
2. Compile circuit and run key generation (`create_prove_verify_keys(...)`)
3. Produce proof (`prove(...)`)
4. Verify proof (`verify(...)`)

---

## 3) Core Programming Model

### `@zk_circuit`

Use `@zk_circuit` to define a top-level circuit function.

- Circuit inputs must be type-annotated.
- Inputs can be marked `Public[...]` or `Private[...]`.
- If `Public`/`Private` is omitted, the compiler defaults the input to `Private`.
- Circuit functions must not return values.
- Input types can be `Integer`, `Float`, `List[dtype0, dtype1, ...]`, `Tuple[dtype0, dtype1, ...]`, or `NDArray[dtype, dim0, dim1, ...]`. Dimensions must be compile-time constants.

Example shape:

```python
@zk_circuit
def my_circuit(
    x: Private[Integer],
    y: Public[Integer],
    out: Public[Boolean],
):
    assert out == (x > y)
```

### `@zk_chip`

Use `@zk_chip` for reusable subroutines called from circuits/chips.

- Chips may return values.
- Chips must have explicit return annotation.
- Chip return types must **not** be wrapped by `Public[...]` / `Private[...]`.

Example shape:

```python
@zk_chip
def score_bucket(x: Integer) -> Integer:
    if x >= 700:
        return 2
    return 1
```

### Data Types and Containers

Common primitives:

- `Integer`, `Float`, `Boolean`
- `NDArray[dtype, dim1, dim2, ...]`
- `Tuple[...]`, `List[...]`

Common privacy wrappers:

- `Public[T]`
- `Private[T]`

---

## 4) DSL Restrictions and Determinism Rules

Zinnia intentionally enforces deterministic circuit shape and type stability.

### Function-level constraints

- Circuits:
    - must have annotated inputs
    - must not declare return annotations
    - do not support `return` statements
- Chips:
    - must declare return annotations (use `-> None` when needed)
    - return annotation cannot be `Public[...]` / `Private[...]`

### Static-inference constraints (important)

The following values must be statically inferable in relevant contexts:

- `range(...)` arguments (`start`, `stop`, `step`) must be integers and statically inferable.
- Generator-expression `if` conditions must be statically inferable.
- Some list/tuple/ndarray operations require statically inferable indices/axes.
- For tuple/list dynamic indexing, heterogeneous element types may be rejected because output type becomes non-deterministic.

### Control-flow constraints

- `break` / `continue` are only valid inside loops.
- `while` loops are bounded by config (`loop_limit`, default: `1000`).
- Recursive chip calls are bounded by config (`recursion_limit`, default: `100`).

### Type/shape stability constraints

- Variables captured from outer scope cannot change datatype/shape in inner scopes.
- `NDArray` annotations require integer dimensions.
- `NDArray` dtype is constrained to supported numeric datatypes.
- Certain slicing/assignment forms require fixed slicing arity.

---

## 5) Operator Coverage (Implemented)

The operator registry is defined in `zinnia/op_def/operator_factory.py`.

| Namespace | Implemented operators |
|---|---|
| Built-in / global | `tuple`, `str`, `range`, `print`, `pow`, `min`, `max`, `list`, `len`, `float`, `bool`, `int`, `sum`, `any`, `all`, `poseidon_hash`, `merkle_verify` |
| `NDArray` methods | `prod`, `sum`, `T`, `transpose`, `tolist`, `astype`, `max`, `min`, `argmax`, `argmin`, `shape`, `reshape`, `flat`, `dtype`, `all`, `any`, `ndim`, `repeat`, `size`, `flatten` |
| `Tuple` methods | `count`, `index` |
| `List` methods | `append`, `extend`, `insert`, `pop`, `copy`, `clear`, `reverse`, `index`, `count`, `remove` |
| `np` / `zinnia` namespace (NumPy-like) | `eye`, `zeros`, `ones`, `identity`, `concatenate`, `concat`, `stack`, `minimum`, `maximum`, `logical_not`, `logical_and`, `logical_or`, `asarray`, `abs`, `absolute`, `acos`, `add`, `all`, `allclose`, `amax`, `amin`, `any`, `argmax`, `argmin`, `array_equal`, `array_equiv`, `asin`, `atan`, `cos`, `cosh`, `divide`, `equal`, `exp`, `fabs`, `floor_divide`, `fmax`, `fmin`, `fmod`, `greater`, `greater_equal`, `isclose`, `less`, `less_equal`, `log`, `logical_xor`, `max`, `mod`, `multiply`, `negative`, `positive`, `not_equal`, `power`, `pow`, `prod`, `sign`, `sinh`, `sqrt`, `subtract`, `sum`, `tan`, `tanh`, `repeat`, `size`, `append`, `dot`, `arange`, `linspace`, `array`, `mean` |
| `math` namespace | `tanh`, `tan`, `log`, `exp`, `cos`, `sin`, `cosh`, `sinh`, `sqrt`, `fabs`, `inv` |

---

## 6) Prove/Verify Lifecycle

The examples follow this sequence:

```python
circuit = create_circuit(my_circuit, chips=[])
keygen_result = create_prove_verify_keys(circuit, proving_data, circuit_name="demo", k=16)
prove_result = prove(circuit, keygen_result, proving_data)
verify_result = verify(keygen_result)
```

### Artifacts produced

Key files generated by the helper pipeline include:

- Rust example source for backend runner (`<circuit_name>.rs`)
- Input payload (`<circuit_name>.in`)
- Proving key (`<circuit_name>.pk`)
- Verifying key (`<circuit_name>.vk`)
- Proof (`<circuit_name>.snark`)

---

## 7) Example Suite (Business-focused)

| Example | Business statement proven |
|---|---|
| `examples/proof_of_solvency.py` | Solvency ratio/pass status from private asset/liability buckets |
| `examples/aml_structuring_detection_proof.py` | AML structuring alert consistency from private transaction pattern |
| `examples/private_credit_policy_check.py` | Credit policy pass/fail with private borrower profile |
| `examples/private_kpi_benchmarking.py` | KPI benchmark outperformance with private operational totals |
| `examples/sanctions_pep_screening.py` | Sanctions/PEP screening outcome consistency with private customer ID |
| `examples/membership_proof.py` | Merkle membership / non-membership statement |

---

## 8) Future Work

This project is a work in progress, and we welcome feedback.

Planned roadmap items include:

1. **Dynamic array operations with zkRAM-backed semantics.** We plan to support non-deterministic array operations where resulting shapes, reduction axes, permutation axes, and slicing indices can be resolved dynamically at proving time instead of being statically inferable at compile time. This will make it easier to model realistic analytics pipelines whose control and indexing decisions depend on private data. The goal is to preserve soundness while reducing DSL friction for variable-shape workloads.

2. **Broader `NDArray` operator coverage.** We plan to expand `NDArray` support to align more closely with common NumPy usage patterns, including additional manipulation, reduction, and convenience operators that are frequently used in analytics code. This should reduce the amount of manual rewrites needed when porting Python/NumPy logic into Zinnia circuits. We will prioritize operators based on practical demand from real proof use cases.

3. **Built-in statistical primitives.** We plan to introduce first-class statistical helpers (for example, distribution-oriented primitives and basic machine learning algorithms) so users can express common analytics assertions more directly in circuits. This will help teams write clearer, less error-prone code without repeatedly implementing the same statistics logic at the circuit layer.