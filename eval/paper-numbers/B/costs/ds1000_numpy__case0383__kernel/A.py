@zk_circuit
def kernel(a: Float[Array, '100'], N: Integer):
    assert N == 7
    N = 7
    result = np.argsort(a)[::-1][:N]
    return result