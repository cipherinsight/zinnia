def kernel(x):
    y=2*x+5
    np.copyto(y,0,where=y<0)
    np.copyto(y,255,where=y>255)
    return y
