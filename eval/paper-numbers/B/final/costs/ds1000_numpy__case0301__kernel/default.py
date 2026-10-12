@zk_circuit
def kernel(A: Float[Array, '20'], ncol: Integer):
    B = np.reshape(A, (-1, ncol))
    return B