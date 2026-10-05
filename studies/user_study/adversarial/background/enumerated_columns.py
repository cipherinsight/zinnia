def kernel(x):
    out = np.empty_like(x)
    for c, column in enumerate(x.T):
        out[:, c] = column - min(column)
    return out
