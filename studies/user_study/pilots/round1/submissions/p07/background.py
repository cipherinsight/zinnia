def kernel(x):
    background = np.min(x, axis=0)
    return x - background
