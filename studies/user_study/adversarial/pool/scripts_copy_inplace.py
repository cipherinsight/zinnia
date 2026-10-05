def kernel(x):
    out = x[:,::2,::2,:].copy()
    out += x[:,1::2,::2,:]
    out += x[:,::2,1::2,:]
    out += x[:,1::2,1::2,:]
    return out
