def kernel(x, y):
    difference = x[:, None, :] - y[None, :, :]
    return np.sum(difference * difference, axis=2)
