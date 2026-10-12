@zk_circuit
def kernel(A: Float[Array, '20'], ncol: Integer):
    assert ncol == 5
    ncol = 5
    B = np.reshape(A, (-1, ncol))
    return B