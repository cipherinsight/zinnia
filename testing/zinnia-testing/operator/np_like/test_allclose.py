import pytest

from zinnia import *

def test_allclose():
    @zk_circuit
    def foo():
        close = np.allclose([1e10,1e-8], [1.00001e10,1e-9])
        assert close
    assert foo()

def test_allclose_false():
    @zk_circuit
    def foo():
        close = np.allclose([1e10,1e-7], [1.00001e10,1e-8])
        assert (not close)
    assert foo()

def test_allclose_false_2():
    @zk_circuit
    def foo():
        close = np.allclose([1e10,1e-8], [1.0001e10,1e-9])
        assert (not close)
    assert foo()

def test_allclose_lhs_numbervalue():
    @zk_circuit
    def foo():
        close = np.allclose(100, [1.00001e10,1e-9])
        assert not close
    assert foo()

def test_allclose_rhs_numbervalue():
    @zk_circuit
    def foo():
        close = np.allclose([1e10,1e-8], 100)
        assert not close
    assert foo()

def test_allclose_atol_tuplevalue():
    @zk_circuit
    def foo():
        close = np.allclose([1e10,1e-8], [1.00001e10,1e-9], atol=(1e-08))
        assert close
    assert foo()

def test_allclose_atol_listvalue():
    @zk_circuit
    def foo():
        close = np.allclose([1e10,1e-8], [1.00001e10,1e-9], atol=[1e-08])
        assert close
    assert foo()

def test_allclose_rtol_tuplevalue():
    @zk_circuit
    def foo():
        close = np.allclose([1e10,1e-8], [1.00001e10,1e-9], rtol=(1e-05))
        assert close
    assert foo()

def test_allclose_rtol_listvalue():
    @zk_circuit
    def foo():
        close = np.allclose([1e10,1e-8], [1.00001e10,1e-9], rtol=[1e-05])
        assert close
    assert foo()

def test_allclose_invalid_lhs():
    @zk_circuit
    def foo():
        close = np.allclose("lhs", [1.00001e10,1e-9])

    with pytest.raises(ZinniaException) as e:
        assert foo()
    assert "Unsupported argument type for `lhs`" in str(e)

def test_allclose_invalid_rhs():
    @zk_circuit
    def foo():
        close = np.allclose([1e10,1e-8], "rhs")

    with pytest.raises(ZinniaException) as e:
        assert foo()
    assert "Unsupported argument type for `rhs`" in str(e)

def test_allclose_invalid_atol():
    @zk_circuit
    def foo():
        close = np.allclose([1e10,1e-8], [1.00001e10,1e-9], atol="atol")

    with pytest.raises(ZinniaException) as e:
        assert foo()
    assert "Unsupported argument type for `atol`" in str(e)

def test_allclose_invalid_rtol():
    @zk_circuit
    def foo():
        close = np.allclose([1e10,1e-8], [1.00001e10,1e-9], rtol="rtol")

    with pytest.raises(ZinniaException) as e:
        assert foo()
    assert "Unsupported argument type for `rtol`" in str(e)

# def test_allclose_nan():
#     @zk_circuit
#     def foo():
#         close = np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
#         assert close
#     assert foo()

# def test_allclose_false_nan():
#     @zk_circuit
#     def foo():
#         close = np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=False)
#         assert (not close)
#     assert foo()