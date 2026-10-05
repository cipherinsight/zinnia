def kernel(x, floor):
    selected = np.where(x > floor, x, 0)
    return np.sum(selected * selected, axis=1)
