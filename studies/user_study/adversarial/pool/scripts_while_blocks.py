def kernel(x):
    b,h,w,c = x.shape
    out = np.zeros((b,h//2,w//2,c), dtype=int)
    i = 0
    while i < h:
        j = 0
        while j < w:
            out[:,i//2,j//2,:] = x[:,i,j,:] + x[:,i+1,j,:] + x[:,i,j+1,:] + x[:,i+1,j+1,:]
            j += 2
        i += 2
    return out
