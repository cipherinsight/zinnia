def kernel(x):
    parts = [x[:,::2,::2,:], x[:,1::2,::2,:], x[:,::2,1::2,:], x[:,1::2,1::2,:]]
    return np.stack(parts,axis=0).sum(axis=0)
