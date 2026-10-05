def kernel(x):
    out=np.empty_like(x)
    for i,v in enumerate(x):
        y=2*v+5
        out[i]=0 if y<0 else 255 if y>255 else y
    return out
