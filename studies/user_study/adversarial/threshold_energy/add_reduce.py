def kernel(x, floor):
    return np.add.reduce(np.where(x > floor, x * x, 0), axis=1)
