def kernel(x):
    return x - np.min(x, axis=(0,))
