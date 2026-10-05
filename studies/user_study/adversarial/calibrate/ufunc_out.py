def kernel(x):
    y=2*x+5
    np.maximum(y,0,out=y)
    np.minimum(y,255,out=y)
    return y
