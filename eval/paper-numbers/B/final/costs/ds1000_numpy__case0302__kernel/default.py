@zk_circuit
def kernel(A: Float[Array, '20'], nrow: Integer):
    B = np.reshape(A, (nrow, -1))
    return B