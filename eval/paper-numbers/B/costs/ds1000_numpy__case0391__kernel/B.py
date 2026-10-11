@zk_circuit
def kernel(a: Float[Array, '20 10'], low: Integer, high: Integer):
    assert low == 4
    low = 4
    assert high == 7
    high = 7
    result = a[:, low:high]
    return result