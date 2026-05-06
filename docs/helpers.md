# Circuit helpers: `@zk_chip` vs `@zk_external` vs no decorator

`@zk_circuit` only captures the source of the function it decorates. Any user-defined
function called from inside a `@zk_circuit` body must therefore be marked so the
compiler can find and lower it. Choose one of the three options below.

## When to use which

- **`@zk_chip`** — the helper participates in the circuit (it is lowered to IR and
  proved). Use this for any pure, deterministic helper called from a
  `@zk_circuit` body. Helpers declared at module scope are auto-discovered by
  `@zk_circuit` from caller locals (see `zinnia/api/zk_circuit.py:182-191`), so
  the only change required is to add the decorator.
- **`@zk_external`** — the helper runs in plain Python during witness
  generation only; it is *not* lowered into the circuit. Use this when the
  helper produces auxiliary data (e.g. setup, sampling, hints) that the
  circuit will then constrain.
- **No decorator** — only safe for module-level constants / classes that are
  never called from a `@zk_circuit` body. Calling a bare `def` from a circuit
  body fails compilation with `Named attribute '.<helper>' not yet implemented`.

## Example

Before (fails to compile, `@zk_circuit` cannot see `relu`):

```python
from zinnia import *

def relu(x):
    return np.maximum(x, 0)

@zk_circuit
def mlp(x: NDArray[Float, 8, 3], w: NDArray[Float, 3, 16], b: NDArray[Float, 16]):
    _zinnia_result = relu(x @ w + b)
```

After (compiles — `relu` is auto-discovered as a chip):

```python
from zinnia import *

@zk_chip
def relu(x):
    return np.maximum(x, 0)

@zk_circuit
def mlp(x: NDArray[Float, 8, 3], w: NDArray[Float, 3, 16], b: NDArray[Float, 16]):
    _zinnia_result = relu(x @ w + b)
```

Recursive helpers (e.g. `fibo`, recursive `partition`) are also valid `@zk_chip`
targets; the compiler unrolls them up to `ZinniaConfig.recursion_limit`.
