"""halo2-base arithmetic agrees with the mock backend and with Python/NumPy.

Covers the fix-halo2base changes end to end through the compiler: integer
`//` and `%` (Python semantics), fixed point `/`, exact `sqrt`,
`sin`/`cos`/`arcsin`, and program-order memory writes. Each case is proved and verified with the separated
setup/prove/verify API. The malicious-witness side (every prover hint
pinned) is tested in Rust, `src/prove/halo2_base/synthesizer_tests.rs`.
"""
import math

import numpy as np
import pytest

from zinnia import *

PARAMS = {"k": 12, "lookup_bits": 8, "precision_bits": 32}
FR = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
SCALE = 2**32


def _decode(hexstr):
    value = int.from_bytes(bytes.fromhex(hexstr), "little")
    return value - FR if value > FR // 2 else value


def _mock_raw(compiled, *args):
    proof = compiled.prove(*args, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"
    return int(proof.public_values[-1].rsplit("=", 1)[1])


def _prove_all(kernel, cases, params=PARAMS):
    """Yield (args, halo2-base raw output, mock raw output) per case."""
    compiled = ZKCircuit.from_method(kernel).compile()
    keys = compiled.setup(params=params)
    for args in cases:
        out = compiled.prove_with_keys(keys, *args)
        assert compiled.verify_with_keys(keys, out), f"verify failed for {args}"
        yield args, _decode(out["public_values"][-1]), _mock_raw(compiled, *args)


def test_integer_floor_division_and_modulo_follow_python():
    @zk_circuit
    def floor_div(a: Integer, b: Integer) -> Integer:
        return a // b

    @zk_circuit
    def modulo(a: Integer, b: Integer) -> Integer:
        return a % b

    cases = [(7, 2), (-7, 2), (7, -2), (-7, -2), (6, 3), (0, 5)]
    for (a, b), got, mock in _prove_all(floor_div, cases):
        assert got == mock == a // b, (a, b, got, mock)
    for (a, b), got, mock in _prove_all(modulo, cases):
        assert got == mock == a % b, (a, b, got, mock)


def test_fixed_point_division_matches_mock_and_numpy():
    @zk_circuit
    def divide(a: Float, b: Float) -> Float:
        return a / b

    for (a, b), got, mock in _prove_all(divide, [(1.0, 3.0), (-1.5, 2.5), (10.0, -4.0), (1e6, 1e-3)]):
        assert got == mock
        assert abs(got / SCALE - a / b) < 1e-6 * max(1.0, abs(a / b))


def test_sqrt_is_the_exact_floor_root():
    @zk_circuit
    def root(x: Float) -> Float:
        return np.sqrt(x)

    for (x,), got, mock in _prove_all(root, [(2.0,), (0.25,), (1e6,), (0.0,)]):
        assert got == mock
        raw = round(x * SCALE)
        assert got >= 0 and got * got <= raw * SCALE < (got + 1) ** 2


def test_trigonometry_matches_mock_and_numpy():
    @zk_circuit
    def sine(x: Float) -> Float:
        return np.sin(x)

    @zk_circuit
    def cosine(x: Float) -> Float:
        return np.cos(x)

    @zk_circuit
    def arcsine(x: Float) -> Float:
        return np.arcsin(x)

    for kernel, reference, xs in [
        (sine, math.sin, [0.5, -1.0, 3.0, 4.0, -7.5]),
        (cosine, math.cos, [0.0, 2.0, -4.0]),
        (arcsine, math.asin, [0.5, -0.3, 1.0, 0.99]),
    ]:
        for (x,), got, mock in _prove_all(kernel, [(v,) for v in xs]):
            assert got == mock, (kernel, x)
            assert abs(got / SCALE - reference(x)) < 2e-7, (kernel, x, got / SCALE)


def test_constant_write_after_dynamic_write_wins():
    """`x[i] = 7; x[0] = 9; return x[0]` with i = 0 proved 7 before (M3).

    The compiler now resolves `x[0]` to the constant 9 (the last write to
    that cell), so the read at a runtime position `j` exercises the order of
    the two writes in the circuit."""

    @zk_circuit
    def overwrite(x: Integer[Array, "1100"], i: Integer) -> Integer:
        x[i] = 7
        x[0] = 9
        return x[0]

    assert [e.get("Int") for e in ZKCircuit.from_method(overwrite).compile().get_elided_publics()] == [9]

    @zk_circuit
    def overwrite_read(x: Integer[Array, "1100"], i: Integer, j: Integer) -> Integer:
        x[i] = 7
        x[0] = 9
        return x[j]

    compiled = ZKCircuit.from_method(overwrite_read).compile()
    # The array's 1,100 cells and its three accesses are rows of the RAM
    # trace (`circuit/ram.rs`), which needs k = 11.
    keys = compiled.setup(params={"k": 11, "lookup_bits": 8, "precision_bits": 32})
    data = np.arange(1100)
    for i, j, expected in ((0, 0, 9), (5, 5, 7), (5, 0, 9), (0, 5, 5)):
        out = compiled.prove_with_keys(keys, data, i, j)
        assert compiled.verify_with_keys(keys, out)
        assert _decode(out["public_values"][-1]) == _mock_raw(compiled, data, i, j) == expected
    # A layout that cannot fit (the trace needs more rows than k has) is an
    # error naming k, not a panic.
    with pytest.raises(RuntimeError, match="k=10"):
        compiled.setup(params={"k": 10, "lookup_bits": 8, "precision_bits": 32})


def test_many_constants_fit_the_constants_columns_at_a_small_k():
    """1,100 constants used to overflow the constants' columns at k=7: the
    layout reserves the blinding rows in every column."""

    @zk_circuit
    def weighted(x: Integer[Array, "1100"]) -> Integer:
        return np.dot(x, np.arange(1100) * 3 + 1)

    compiled = ZKCircuit.from_method(weighted).compile()
    small = compiled.setup(params={"k": 7, "lookup_bits": 6, "precision_bits": 32})
    data = np.arange(1100) % 7
    out = compiled.prove_with_keys(small, data)
    assert compiled.verify_with_keys(small, out)
    assert _decode(out["public_values"][-1]) == _mock_raw(compiled, data) == int(np.dot(data, np.arange(1100) * 3 + 1))
    with pytest.raises(RuntimeError, match="k=6"):
        compiled.setup(params={"k": 6, "lookup_bits": 8, "precision_bits": 32})


def test_separated_steps_reuse_keys_and_reject_tampering():
    @zk_circuit
    def affine(a: Float, b: Float) -> Float:
        return a * b + 1.0

    compiled = ZKCircuit.from_method(affine).compile()
    keys = compiled.setup(params=PARAMS)
    assert (keys.k, keys.lookup_bits, keys.precision_bits) == (12, 8, 32)
    assert keys.keygen_seconds > 0 and len(keys.vk_bytes()) > 0
    first = compiled.prove_with_keys(keys, 1.5, 2.0)
    second = compiled.prove_with_keys(keys, -3.0, 0.25)
    for out in (first, second):
        assert compiled.verify_with_keys(keys, out)
        assert out["witness_seconds"] >= 0 and out["prove_seconds"] > 0
    assert _decode(first["public_values"][-1]) == round(4.0 * SCALE)

    tampered = dict(first, public_values=list(first["public_values"]))
    tampered["public_values"][-1] = second["public_values"][-1]
    assert not compiled.verify_with_keys(keys, tampered)
    assert not compiled.verify_with_keys(keys, dict(first, proof=second["proof"]))

    @zk_circuit
    def other(a: Float, b: Float) -> Float:
        return a - b

    other_compiled = ZKCircuit.from_method(other).compile()
    with pytest.raises(ValueError, match="different circuit"):
        other_compiled.prove_with_keys(keys, 1.0, 2.0)
