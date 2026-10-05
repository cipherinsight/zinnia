def kernel(x, y):
    delta = x[:,None,:] - y[None,:,:]
    np.square(delta,out=delta)
    result = np.zeros((len(x),len(y)),dtype=int)
    result = np.sum(delta,axis=2)
    return result
