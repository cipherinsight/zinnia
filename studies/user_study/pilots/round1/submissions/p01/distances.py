def kernel(x, y):
    delta = x[:, None, :] - y[None, :, :]
    return np.sum(delta * delta, axis=2)
