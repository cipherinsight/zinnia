def kernel(x):
    low = np.empty(x.shape[1], dtype=x.dtype)
    np.min(x, axis=0, out=low)
    return x - low
