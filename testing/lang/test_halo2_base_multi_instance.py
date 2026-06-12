"""C17: halo2-base multi-instance copy-constraint regression.

`build_circuit` first tries a layout-only fast path
(`try_build_public_layout_only`) that handles circuits whose public values
are read straight from inputs and forwarded to the instance column with no
intervening computation. When that fast path bails (any real op appears) it
returns `None` and `build_circuit` re-runs the FULL synthesizer on the SAME
`BaseCircuitBuilder`.

The bug: the fast path pushed each public-input cell into the shared
`builder.assigned_instances` as it walked the IR, and those pushes were NOT
rolled back when it bailed. The full synthesizer then exposed the same
public inputs again, so `assigned_instances` ended up LONGER than the
`public_values` vector that becomes the instance column. halo2-base emits
one copy constraint per `assigned_instances` entry targeting instance row
`i`; the surplus entry pointed past the end of the instance column, so
`verify()` (and MockProver) failed with a constraint-system error for ANY
circuit exposing a public input alongside a computed public output.

This only fires when a circuit has at least one `Public[...]` INPUT and also
does computation (so the fast path bails). DS-1000 kernels take private
inputs and expose only outputs, so they never tripped it — which is why the
134-case four-way benchmark verified clean despite the bug.

Shapes (a)-(c) failed `verify()` at HEAD before the fix; (d) is the
single-instance no-regression guard that always worked (pure layout path).
"""
import pytest

from zinnia import *

HALO2_PARAMS = {"k": 12, "precision_bits": 32, "lookup_bits": 8}

_BN256_FR = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001


def _decode_field_int(hexstr):
    val = int.from_bytes(bytes.fromhex(hexstr), "little")
    if val > _BN256_FR // 2:
        val -= _BN256_FR
    return val


def test_public_input_plus_output_two_instances():
    """(a) `v + 1`: a public input and a computed public output."""

    @zk_circuit
    def vp1(v: Public[Integer]) -> Integer:
        return v + 1

    circuit = ZKCircuit.from_method(vp1)
    proof = circuit.prove(5, backend="halo2-base", params=HALO2_PARAMS)
    assert circuit.verify(proof), "v+1 (public in + public out) must verify"
    vals = [_decode_field_int(h) for h in proof.public_values]
    assert vals == [5, 6], vals


def test_two_public_inputs_plus_output_three_instances():
    """(b) `a + b`: two public inputs and a computed public output."""

    @zk_circuit
    def apb(a: Public[Integer], b: Public[Integer]) -> Integer:
        return a + b

    circuit = ZKCircuit.from_method(apb)
    proof = circuit.prove(3, 4, backend="halo2-base", params=HALO2_PARAMS)
    assert circuit.verify(proof), "a+b (2 public in + public out) must verify"
    vals = [_decode_field_int(h) for h in proof.public_values]
    assert vals == [3, 4, 7], vals


def test_three_plus_instances_with_repeated_value():
    """(c) 3+ instances including a repeated public value.

    `a`, `b`, and `a + b - b == a` are exposed: the third public value
    equals the first, exercising the copy-constraint path for a duplicated
    instance value across distinct advice cells.
    """

    @zk_circuit
    def repeat(a: Public[Integer], b: Public[Integer]) -> Integer:
        return a + b - b

    circuit = ZKCircuit.from_method(repeat)
    proof = circuit.prove(6, 9, backend="halo2-base", params=HALO2_PARAMS)
    assert circuit.verify(proof), "repeated-value multi-instance must verify"
    vals = [_decode_field_int(h) for h in proof.public_values]
    assert vals == [6, 9, 6], vals


def test_single_instance_output_only_no_regression():
    """(d) Private input, single public output — the pure-layout fast path
    that always worked. Guards against a regression from the fix."""

    @zk_circuit
    def single(v: Integer) -> Integer:
        return v + 1

    circuit = ZKCircuit.from_method(single)
    proof = circuit.prove(41, backend="halo2-base", params=HALO2_PARAMS)
    assert circuit.verify(proof), "single public output must verify"
    vals = [_decode_field_int(h) for h in proof.public_values]
    assert vals == [42], vals
