def kernel(x, y):
    delta = x[:, None, :] - y[None, :, :]
    return np.sum(delta ** 2, axis=-1)
