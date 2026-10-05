def kernel(x, floor):
    return np.sum(x ** 2, axis=1, where=x > floor, initial=0)
