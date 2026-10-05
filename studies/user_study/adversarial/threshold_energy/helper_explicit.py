def kernel(x, floor):
    def energy(row, limits):
        total = 0
        for v, f in zip(row, limits):
            if v > f:
                total += v ** 2
        return total
    return np.array([energy(row, floor) for row in x])
