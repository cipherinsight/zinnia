def kernel(x, floor):
    return np.where(x > floor, x * x, 0) @ np.ones(x.shape[1], dtype=int)
