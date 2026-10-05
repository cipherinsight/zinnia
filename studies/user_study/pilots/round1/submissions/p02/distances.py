def kernel(x, y):
    difference = x[:, None, :] - y[None, :, :]
    return (difference * difference).sum(axis=2)
