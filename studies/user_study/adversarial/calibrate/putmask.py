def kernel(x):
    y=2*x+5
    np.putmask(y,y<0,0)
    np.putmask(y,y>255,255)
    return y
