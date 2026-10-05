def kernel(x):
    out = x.copy()
    out -= x.min(axis=0)
    return out
