def kernel(x, floor):
    return ((x > floor) * x * x).sum(axis=-1)
