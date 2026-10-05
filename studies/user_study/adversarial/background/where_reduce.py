def kernel(x):
    low = x[0].copy()
    for row in x[1:]:
        low = np.where(row < low, row, low)
    return np.subtract(x, low)
