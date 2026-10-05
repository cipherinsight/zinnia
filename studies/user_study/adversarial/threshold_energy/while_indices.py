def kernel(x, floor):
    result = np.zeros(x.shape[0], dtype=int)
    n = 0
    while n < x.shape[0]:
        c = 0
        total = 0
        while c < x.shape[1]:
            v = x[n, c]
            total += v * v if v > floor[c] else 0
            c += 1
        result[n] = total
        n += 1
    return result
