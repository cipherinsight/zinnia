def kernel(x, floor):
    return np.sum(np.where(x > floor, x ** 2, 0), axis=(1,), keepdims=True).squeeze(axis=1)
