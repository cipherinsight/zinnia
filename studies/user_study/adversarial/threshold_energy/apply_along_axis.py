def kernel(x, floor):
    def energy(row):
        return np.sum(np.where(row > floor, row * row, 0))
    return np.apply_along_axis(energy, 1, x)
