@zk_circuit
def kernel(A: Float[Array, '20'], nrow: Integer):
    assert nrow == 5
    nrow = 5
    B = np.reshape(A, (nrow, -1))
    return B