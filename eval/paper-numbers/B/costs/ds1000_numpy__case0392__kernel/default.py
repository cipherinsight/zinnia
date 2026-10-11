@zk_circuit
def kernel(a: Float[Array, '20 10'], low: Integer, high: Integer):
    result = a[low:high, :]
    return result