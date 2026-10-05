def kernel(x):
    y=2*x+5
    return np.select([y<0,y>255],[0,255],default=y)
