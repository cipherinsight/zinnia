@zk_circuit
def kernel(a: Float[Array, '6 3 4'], h: Integer, w: Integer):
    assert h == 6
    h = 6
    assert w == 12
    w = 12
    n, nrows, ncols = a.shape
    result = a.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)
    return result