def kernel(x):
    return x - np.sort(x, axis=0)[0]
