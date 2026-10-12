@zk_circuit
def kernel(A: Float[Array, '23'], ncol: Integer):
    col = A.shape[0] // ncol * ncol
    B = A[:col]
    B = np.reshape(B, (-1, ncol))
    return B