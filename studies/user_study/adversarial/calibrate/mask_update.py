def kernel(x):
    y=2*x+5
    y[y < 0]=0
    y[y > 255]=255
    return y
