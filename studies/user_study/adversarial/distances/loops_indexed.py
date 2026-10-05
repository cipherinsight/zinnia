def kernel(x, y):
    n, d = x.shape
    m = y.shape[0]
    result = np.zeros((n,m), dtype=int)
    for i in range(n):
        for j in range(m):
            value = 0
            for k in range(d):
                diff = x[i,k] - y[j,k]
                value += diff * diff
            result[i,j] = value
    return result
