# Helper functions in circuits

`@zk_circuit` reads only the source of the function it decorates. A function that the circuit
calls must therefore be one the compiler can see: a function marked `@zk_chip` or
`@zk_external`, or a function defined inside the circuit body.

## The three kinds of helper

- **`@zk_chip`**: the helper is part of the circuit. It is compiled like the circuit body and
  inlined at each call, so its result is constrained. Use it for any deterministic helper.
  `@zk_circuit` finds the chips (and external functions) defined in the circuit's module and in
  the scope where the circuit is called (`_with_registered_functions` in
  `zinnia/api/zk_circuit.py`), so adding the decorator is the only change needed. Parameter and
  return annotations are optional (types are taken from the call site). A recursive chip is
  inlined until its recursive call is proved unreachable, at most
  `ZinniaConfig.recursion_limit` calls deep (default 16); a deeper call is a compile error.
- **`@zk_external`**: the helper runs in plain Python while the witness is generated and is not
  compiled into the circuit. Its result enters the circuit as a witness value that no constraint
  ties to the helper's computation, so the circuit must check whatever it relies on (for
  example, a hint that the circuit then verifies). It needs a return annotation
  (`def f(...) -> Integer:`).
- **A function defined inside the circuit body** (a nested `def` or a lambda) is turned into a
  chip by the compiler (`zinnia/compile/lowering/nested_defs.py`,
  `zinnia/compile/lowering/helpers.py`), including one that reads the circuit's variables. The
  module docstring of `helpers.py` lists the forms it rejects (recursion, a helper defined in a
  branch, `nonlocal`, decorators, and others).

A plain module-level function without a decorator is not visible. Calling it from a circuit is a
compile error:

```
Unknown function `relu` in @zk_circuit. If `relu` is a helper you defined, decorate it with
@zk_chip (or @zk_external for off-circuit witness functions). ...
```

## Example

This fails to compile, because `relu` is undecorated and defined outside the circuit:

```python
from zinnia import *

def relu(x):
    return np.maximum(x, 0)

@zk_circuit
def mlp(x: NDArray[Float, 8, 3], w: NDArray[Float, 3, 16], b: NDArray[Float, 16]):
    return relu(x @ w + b)
```

With the decorator, `@zk_circuit` finds `relu` as a chip and the circuit compiles:

```python
from zinnia import *

@zk_chip
def relu(x):
    return np.maximum(x, 0)

@zk_circuit
def mlp(x: NDArray[Float, 8, 3], w: NDArray[Float, 3, 16], b: NDArray[Float, 16]):
    return relu(x @ w + b)
```

Tests: `testing/lang/test_chip_helpers.py`, `testing/lang/test_unknown_call_diagnostic.py`,
`testing/api/test_external_func.py`.
