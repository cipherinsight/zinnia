"""Complex transcendentals: np.exp, np.sin, np.cos, np.sqrt over Value::Complex.

Built on the existing real-Float trig + exp + sqrt gates via Euler-style
identities. Vectorized over arrays through `vectorize_unary_np`.
"""
from zinnia import zk_circuit, ZKCircuit, np


def test_complex_exp_pure_imag_compiles():
    """exp(i*pi) ≈ -1 (Euler's identity)."""
    @zk_circuit
    def foo():
        z = np.exp(1j * np.pi)
        # Don't assert exact equality (precision); just compile + use.
        _ = z.real
        _ = z.imag

    ZKCircuit.from_method(foo).compile()


def test_complex_exp_real_argument_compiles():
    """exp(2 + 0j) = e^2 + 0i."""
    @zk_circuit
    def foo():
        z = np.exp(2 + 0j)
        _ = z

    ZKCircuit.from_method(foo).compile()


def test_complex_sin_compiles():
    @zk_circuit
    def foo():
        z = np.sin(0 + 1j)  # ≈ i*sinh(1)
        _ = z

    ZKCircuit.from_method(foo).compile()


def test_complex_cos_compiles():
    @zk_circuit
    def foo():
        z = np.cos(0 + 1j)  # ≈ cosh(1)
        _ = z

    ZKCircuit.from_method(foo).compile()


def test_complex_sqrt_real_input():
    @zk_circuit
    def foo():
        z = np.sqrt(4 + 0j)  # = 2 + 0i
        _ = z

    ZKCircuit.from_method(foo).compile()


def test_complex_sqrt_pure_imag():
    @zk_circuit
    def foo():
        z = np.sqrt(0 + 4j)  # = sqrt(2)*(1 + i)
        _ = z

    ZKCircuit.from_method(foo).compile()


def test_complex_exp_over_array():
    @zk_circuit
    def foo():
        a = np.zeros((3,), dtype=complex)
        a[0] = 0 + 0j
        a[1] = 0 + 1j
        a[2] = 1 + 1j
        b = np.exp(a)
        _ = b[0].real
        _ = b[2].imag

    ZKCircuit.from_method(foo).compile()
