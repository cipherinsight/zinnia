def kernel(x):
    return np.stack([column - min(column) for column in x.T], axis=1)
