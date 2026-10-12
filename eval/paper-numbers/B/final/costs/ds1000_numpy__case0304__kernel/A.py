@zk_circuit
def kernel(A: Float[Array, '23'], ncol: Integer):
    assert ncol == 5
    ncol = 5
    col = A.shape[0] // ncol * ncol
    B = A[len(A) - col:][::-1]
    B = np.reshape(B, (-1, ncol))
    return B