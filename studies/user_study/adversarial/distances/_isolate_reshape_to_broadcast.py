def kernel(x, y):
    n, d = x.shape
    m, _ = y.shape
    delta = x[:,None,:]-y[None,:,:]
    return np.sum(np.square(delta), axis=2)
