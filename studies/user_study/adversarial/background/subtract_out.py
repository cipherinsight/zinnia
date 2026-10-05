def kernel(x):
    out = np.empty_like(x)
    np.subtract(x, x.min(axis=0), out=out)
    return out
