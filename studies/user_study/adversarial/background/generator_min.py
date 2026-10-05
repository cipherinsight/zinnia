def kernel(x):
    low = np.array([min(x[t, c] for t in range(x.shape[0])) for c in range(x.shape[1])])
    return x - low
