def kernel(x, floor):
    active = x.copy()
    active[x <= floor] = 0
    return (active ** 2).sum(axis=1)
