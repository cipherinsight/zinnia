@zk_circuit
def kernel(a: Float[Array, '100'], N: Integer):
    result = np.argsort(a)[::-1][:N]
    return result