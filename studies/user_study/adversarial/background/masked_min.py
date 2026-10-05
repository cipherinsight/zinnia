def kernel(x):
    return x - np.min(x, axis=0, where=np.ones_like(x, dtype=bool), initial=4096)
