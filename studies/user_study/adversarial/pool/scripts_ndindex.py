def kernel(x):
    b,h,w,c = x.shape
    out = np.zeros((b,h//2,w//2,c),dtype=int)
    for n,i,j,k in np.ndindex(out.shape):
        out[n,i,j,k] = x[n,2*i,2*j,k] + x[n,2*i+1,2*j,k] + x[n,2*i,2*j+1,k] + x[n,2*i+1,2*j+1,k]
    return out
