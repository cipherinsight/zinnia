def kernel(x, y):
    result = np.zeros((x.shape[0], y.shape[0]), dtype=int)
    for k in range(x.shape[1]):
        delta = x[:,k,None] - y[None,:,k]
        result += delta*delta
    return result
