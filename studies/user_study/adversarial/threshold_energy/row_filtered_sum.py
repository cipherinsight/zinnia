def kernel(x, floor):
    return np.array([np.sum(row[row > floor] ** 2) for row in x])
