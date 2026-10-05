def kernel(x, y):
    delta = np.subtract(x[:,None,:],y[None,:,:])
    return np.add.reduce(np.multiply(delta,delta),axis=2)
