@zk_circuit
def kernel(a: Float[Array, '20 10'], low: Integer, high: Integer):
    high = min(high, a.shape[1])
    result = a[:, low:high]
    return result