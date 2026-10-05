def kernel(x, y):
    delta = x[:,None,:] - y[None,:,:]
    return sum(np.moveaxis(delta*delta,2,0))
