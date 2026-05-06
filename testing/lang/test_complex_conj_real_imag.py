"""np.conj, .real, .imag, abs(z) on Complex."""
from zinnia import zk_circuit, ZKCircuit, np


def test_complex_real():
    @zk_circuit
    def foo():
        z = 3 + 4j
        assert z.real == 3.0

    assert foo()


def test_complex_imag():
    @zk_circuit
    def foo():
        z = 3 + 4j
        assert z.imag == 4.0

    assert foo()


def test_complex_conjugate_method():
    @zk_circuit
    def foo():
        z = 3 + 4j
        c = z.conjugate()
        assert c.real == 3.0
        assert c.imag == -4.0

    assert foo()


def test_np_conj():
    @zk_circuit
    def foo():
        z = 1 + 2j
        c = np.conj(z)
        assert c.real == 1.0
        assert c.imag == -2.0

    assert foo()


def test_np_conjugate_alias():
    @zk_circuit
    def foo():
        z = 5 + 6j
        c = np.conjugate(z)
        assert c.imag == -6.0

    assert foo()


def test_abs_complex():
    @zk_circuit
    def foo():
        z = 3 + 4j
        m = abs(z)
        # |3 + 4i| = 5
        assert m == 5.0

    assert foo()


def test_abs_pure_imag():
    @zk_circuit
    def foo():
        z = 0 + 5j
        assert abs(z) == 5.0

    assert foo()


def test_abs_zero_complex():
    @zk_circuit
    def foo():
        z = 0 + 0j
        assert abs(z) == 0.0

    assert foo()
