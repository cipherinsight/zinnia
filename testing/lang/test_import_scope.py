"""Imports outside NumPy and `math` are out of scope (audit finding X7).

The allowed set lives in `zinnia.compile.lowering.imports.ALLOWED_IMPORT_ROOTS`.
"""
import pathlib
import textwrap

import numpy as np
import pytest

from zinnia import ZKChip, ZKCircuit
from zinnia.compile.lowering import ALLOWED_IMPORT_ROOTS
from zinnia.debug.exception import ZinniaException


def accepts(program, *args):
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


def test_allowed_set_is_numpy_and_math():
    assert ALLOWED_IMPORT_ROOTS == {"numpy", "math"}


@pytest.mark.parametrize("statement,module", [
    ("import copy", "copy"),
    ("import itertools as it", "itertools"),
    ("from functools import reduce", "functools"),
    ("import cmath", "cmath"),
    ("from scipy.special import erf", "scipy.special"),
])
def test_out_of_scope_import_in_circuit(statement, module):
    source = f'''def kernel(x: Integer[Array, "3"]):
    {statement}
    return x
'''
    with pytest.raises(ZinniaException, match=f"out of scope: imports {module}"):
        ZKCircuit.from_source("kernel", source).compile()


def test_out_of_scope_import_in_chip():
    with pytest.raises(ZinniaException, match="out of scope: imports copy"):
        ZKChip.from_source("helper", "def helper(a: Integer):\n    import copy\n    return a\n")


@pytest.mark.parametrize("statement,call", [
    ("import math", "math.sqrt(x[0])"),
    ("from math import sqrt", "sqrt(x[0])"),
    ("import numpy", "numpy.sqrt(x[0])"),
    ("from numpy import sqrt as root", "root(x[0])"),
])
def test_allowed_imports_compile_and_compute(statement, call):
    source = f'''def kernel(x: Float[Array, "2"], expected: Float):
    {statement}
    assert {call} == expected
'''
    program = ZKCircuit.from_source("kernel", source).compile()
    assert accepts(program, np.array([4.0, 1.0]), 2.0)
    assert not accepts(program, np.array([4.0, 1.0]), 3.0)


def test_module_level_import_used_by_circuit_is_checked(tmp_path, monkeypatch):
    module = tmp_path / "kernel_with_itertools.py"
    module.write_text(textwrap.dedent('''
        from zinnia import *
        import itertools as it

        @zk_circuit
        def kernel(x: Integer[Array, "3"]):
            return np.array(list(it.accumulate(x)))
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ZinniaException, match="out of scope: imports itertools"):
        __import__("kernel_with_itertools")


MODULE_LEVEL_IMPORTS = [
    ("import numpy", "numpy.sqrt(x[0]) + numpy.sum(x)"),
    ("import numpy as xp", "xp.sqrt(x[0]) + xp.sum(x)"),
    ("import numpy.linalg", "numpy.sqrt(x[0]) + numpy.sum(x)"),
    ("from numpy import sqrt, sum as total", "sqrt(x[0]) + total(x)"),
    ("import math\nimport numpy", "math.sqrt(x[0]) + numpy.sum(x)"),
    ("from math import sqrt as root\nimport numpy", "root(x[0]) + numpy.sum(x)"),
]


@pytest.mark.parametrize("index", range(len(MODULE_LEVEL_IMPORTS)))
def test_module_level_numpy_and_math_imports_resolve(index, tmp_path, monkeypatch):
    # The decorator sees only the function's source; a module-level
    # `import numpy` used to leave `numpy.sqrt` unresolved ("Named attribute
    # `numpy.sqrt` not yet implemented").
    statements, expression = MODULE_LEVEL_IMPORTS[index]
    name = f"kernel_module_imports_{index}"
    (tmp_path / f"{name}.py").write_text(
        "from zinnia import *\n" + statements + "\n\n"
        "@zk_circuit\n"
        'def kernel(x: Float[Array, "2"], expected: Float):\n'
        f"    assert {expression} == expected\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    program = ZKCircuit.from_method(__import__(name).kernel).compile()
    assert accepts(program, np.array([4.0, 1.0]), 7.0)
    assert not accepts(program, np.array([4.0, 1.0]), 8.0)


def test_module_level_imports_resolve_in_chips_and_respect_local_names(tmp_path, monkeypatch):
    (tmp_path / "kernel_module_imports_chip.py").write_text(textwrap.dedent('''
        from zinnia import *
        import numpy
        from math import sqrt

        @zk_chip
        def norm1(v: Float[Array, "2"]) -> Float:
            return numpy.sum(numpy.abs(v))

        @zk_circuit
        def kernel(x: Float[Array, "2"], expected: Float):
            sqrt = 3.0
            assert norm1(x) + sqrt == expected
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    program = ZKCircuit.from_method(__import__("kernel_module_imports_chip").kernel).compile()
    assert accepts(program, np.array([-4.0, 1.0]), 8.0)
    assert not accepts(program, np.array([-4.0, 1.0]), 7.0)
