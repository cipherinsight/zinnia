def kernel(x):
    b,h,w,c = x.shape
    def block(i,j):
        return x[:,i,j,:] + x[:,i+1,j,:] + x[:,i,j+1,:] + x[:,i+1,j+1,:]
    out = np.zeros((b,h//2,w//2,c),dtype=int)
    for i in range(h//2):
        for j in range(w//2):
            out[:,i,j,:] = block(2*i,2*j)
    return out
