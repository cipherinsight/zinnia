def kernel(x, y):
    xx = np.sum(x ** 2, axis=1)
    yy = np.sum(y ** 2, axis=1)
    return xx[:, None] + yy[None, :] - 2 * np.dot(x, np.transpose(y))
