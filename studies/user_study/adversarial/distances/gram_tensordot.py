def kernel(x, y):
    return np.sum(x*x, axis=1)[:,None] + np.sum(y*y, axis=1)[None,:] - 2*np.tensordot(x,y,axes=([1],[1]))
