@zk_circuit
def kernel(a: Integer[Array, '4 4'], size: Integer[Array, '2']):
    result = window(a, size)
    return result