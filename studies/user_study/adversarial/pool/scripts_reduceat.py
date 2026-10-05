def kernel(x):
    y = np.add.reduceat(x, np.arange(0,x.shape[1],2), axis=1)
    return np.add.reduceat(y, np.arange(0,x.shape[2],2), axis=2)
