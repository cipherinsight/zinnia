def kernel(x, y):
    delta = x[:, None, :] - y[None, :, :]
    return (delta * delta).sum(axis=2)
