@zk_circuit
def kernel(a: Integer[Array, '4 4'], size: Integer[Array, '2']):
    assert size[0] == 3
    assert size[1] == 3
    size = np.array([3, 3])
    result = window(a, size)
    return result