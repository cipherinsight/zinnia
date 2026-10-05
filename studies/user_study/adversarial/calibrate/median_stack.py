def kernel(x):
    y=2*x+5
    return np.median(np.stack([np.zeros_like(x),y,np.full_like(x,255)]),axis=0).astype(int)
