def kernel(x):
    b,h,w,c = x.shape
    out = np.zeros((b,h//2,w//2,c), dtype=int)
    for i in range(h//2):
        for j in range(w//2):
            out[:,i,j,:] = np.sum(x[:,2*i:2*i+2,2*j:2*j+2,:], axis=(1,2))
    return out
