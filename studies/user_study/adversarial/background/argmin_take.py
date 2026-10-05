def kernel(x):
    idx = np.argmin(x, axis=0)
    low = x[idx, np.arange(x.shape[1])]
    return x - low
