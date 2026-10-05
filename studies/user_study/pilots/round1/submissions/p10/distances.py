def kernel(x, y):
    differences = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    return np.sum(differences * differences, axis=2)
