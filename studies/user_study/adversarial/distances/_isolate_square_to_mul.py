def kernel(x, y):
    n, d = x.shape
    m, _ = y.shape
    delta = x.reshape(n, 1, d) - y.reshape(1, m, d)
    return np.sum(delta*delta, axis=2)
