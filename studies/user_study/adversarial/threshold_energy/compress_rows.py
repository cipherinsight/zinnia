def kernel(x, floor):
    return np.array([np.sum(np.compress(row > floor, row) ** 2) for row in x])
