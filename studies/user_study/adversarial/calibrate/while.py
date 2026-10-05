def kernel(x):
    out=np.zeros_like(x)
    i=0
    while i<len(x):
        out[i]=min(255,max(0,2*x[i]+5))
        i+=1
    return out
