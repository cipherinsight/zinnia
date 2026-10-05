def kernel(x):
    y=2*x+5
    return np.piecewise(y, [y<0, y>255], [0,255,lambda v:v])
