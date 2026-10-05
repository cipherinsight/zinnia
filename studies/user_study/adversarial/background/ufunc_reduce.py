def kernel(x):
    return x - np.minimum.reduce(x, axis=0)
