def kernel(x, y):
    differences = x[:, None, :] - y[None, :, :]
    return np.sum(differences * differences, axis=2)
