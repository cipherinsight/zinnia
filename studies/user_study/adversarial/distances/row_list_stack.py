def kernel(x, y):
    return np.stack([np.sum((y-a)**2, axis=1) for a in x], axis=0)
