def kernel(x, floor):
    return np.where(x > floor, x * x, 0).sum(axis=1)
