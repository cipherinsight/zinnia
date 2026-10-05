def kernel(x, y):
    n,d = x.shape
    m = y.shape[0]
    a = np.repeat(x,m,axis=0)
    b = np.tile(y,(n,1))
    return np.sum((a-b)**2,axis=1).reshape(n,m)
