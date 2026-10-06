"""`>>` and `<<` on int64 prove NumPy's values on the mock, halo2-base and Noir (npb-gaps).

Before, neither circuit backend had a shift ("does not yet support shr_i"), and
the mock clamped the count to [0, 63]: `x >> -1` gave `x` and `x << 64` gave
`x << 63`. NumPy's int64 shifts (`npy_rshift` / `npy_lshift`) read the count
as an unsigned 64-bit value: `>>` is arithmetic (`floor(x / 2^c)`), and any
count outside [0, 64), negative ones included, gives the sign fill (0 or -1)
for `>>` and 0 for `<<`. That is now `kernel::fp_shr` / `fp_shl`, and every
backend computes it. A `>>` operand or a count outside the int64 window has
no proof (the shift does not commute with int64 wrap-around).
"""
import numpy as np
import pytest

from testing.lang.all_backends import (
    assert_all_backends_match_numpy,
    halo2_field_publics,
    nargo_available,
    noir_field_publics,
)
from testing.lang.mock_outputs import compile_kernel

A = np.array([5, -5, 12345, -12345, 2**62 + 3, -(2**63), 2**63 - 1, 0, 1, -1, 255, 65535], dtype=np.int64)
COUNTS = np.array([1, 1, 7, 7, 63, 1, 62, 64, 100, -1, -64, 0], dtype=np.int64)


def test_dynamic_counts():
    source = 'def kernel(a: Integer[Array, "12"], b: Integer[Array, "12"]):\n    return a >> b, a << b\n'
    assert_all_backends_match_numpy(source, A, COUNTS)


@pytest.mark.parametrize("count", [0, 1, 8, 63, 64, 70, -1])
def test_constant_counts(count):
    source = f'def kernel(a: Integer[Array, "12"]):\n    return a >> {count}, a << {count}\n'
    if count < 0:  # NumPy's scalar shift by a negative Python int is an array shift by int64.
        source = f'def kernel(a: Integer[Array, "12"]):\n    return a >> np.int64({count}), a << np.int64({count})\n'
    assert_all_backends_match_numpy(source, A)


def test_crc16_style_loop():
    """The npbench crc16 kernel's shifts and masks."""
    source = (
        'def kernel(data: Integer[Array, "4"], poly: Integer):\n'
        '    crc = 65535\n'
        '    for b in data:\n'
        '        cur_byte = 255 & b\n'
        '        for _ in range(0, 8):\n'
        '            if crc & 1 ^ cur_byte & 1:\n'
        '                crc = crc >> 1 ^ poly\n'
        '            else:\n'
        '                crc >>= 1\n'
        '            cur_byte >>= 1\n'
        '    crc = ~crc & 65535\n'
        '    crc = crc << 8 | crc >> 8 & 255\n'
        '    return crc & 65535\n'
    )
    assert_all_backends_match_numpy(source, np.array([0x31, 0x32, 0xFF, 0x00], dtype=np.int64), 33800)


@pytest.mark.parametrize("source", [
    # A sum leaving the int64 window, shifted right: NumPy wraps first.
    'def kernel(a: Integer, b: Integer):\n    return (a + b) >> 1\n',
    'def kernel(a: Integer, b: Integer):\n    return 1 << (a + b)\n',
])
def test_operands_outside_the_window_have_no_proof(source):
    program = compile_kernel(source)
    args = (2**63 - 1, 2**63 - 1)
    assert program.prove(*args, backend="mock").proof_bytes_hex != "mock_satisfied"
    with pytest.raises(Exception):
        halo2_field_publics(program, *args)
    if nargo_available():
        with pytest.raises(Exception):
            noir_field_publics(program, *args)
