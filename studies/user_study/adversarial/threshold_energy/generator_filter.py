def kernel(x, floor):
    return np.array([sum(v * v for v, f in zip(row, floor) if v > f) for row in x])
