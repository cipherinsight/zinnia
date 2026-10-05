def kernel(x, floor):
    return np.sum(np.where(x > floor, x * x, 0), axis=1)
