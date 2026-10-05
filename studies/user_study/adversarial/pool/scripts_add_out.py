def kernel(x):
    y = np.zeros_like(x[:,::2,:,:])
    np.add(x[:,::2,:,:],x[:,1::2,:,:],out=y)
    out = np.zeros_like(y[:,:,::2,:])
    np.add(y[:,:,::2,:],y[:,:,1::2,:],out=out)
    return out
