def kernel(x, floor):
    return np.array(list(map(lambda row: np.sum(np.where(row > floor, row * row, 0)), x)))
