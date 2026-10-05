def kernel(x, y):
    xx = np.sum(x * x, axis=1, keepdims=True)
    yy = np.sum(y * y, axis=1, keepdims=True)
    return xx + yy.T - 2 * (x @ y.T)
