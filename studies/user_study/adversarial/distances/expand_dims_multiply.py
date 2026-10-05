def kernel(x, y):
    delta = np.expand_dims(x, 1) - np.expand_dims(y, 0)
    return (delta * delta).sum(axis=2)
