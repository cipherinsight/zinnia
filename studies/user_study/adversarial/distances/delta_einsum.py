def kernel(x, y):
    delta = x[:, None, :] - y[None, :, :]
    return np.einsum('ijd,ijd->ij', delta, delta)
