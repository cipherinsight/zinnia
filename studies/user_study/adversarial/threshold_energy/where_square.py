def kernel(x, floor):
    active = np.where(x > floor, x, 0)
    return np.square(active).sum(axis=1)
