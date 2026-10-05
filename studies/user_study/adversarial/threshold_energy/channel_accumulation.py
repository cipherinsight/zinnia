def kernel(x, floor):
    result = np.zeros(x.shape[0], dtype=int)
    for c, f in enumerate(floor):
        values = x[:, c]
        result += np.where(values > f, values ** 2, 0)
    return result
