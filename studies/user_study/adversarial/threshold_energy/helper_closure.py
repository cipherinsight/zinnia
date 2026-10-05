def kernel(x, floor):
    def energy(row):
        return sum(v * v for v, f in zip(row, floor) if v > f)
    return np.array([energy(row) for row in x])
