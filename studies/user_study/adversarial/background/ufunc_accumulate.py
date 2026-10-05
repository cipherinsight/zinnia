def kernel(x):
    return x - np.minimum.accumulate(x, axis=0)[-1]
