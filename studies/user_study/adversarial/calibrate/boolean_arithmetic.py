def kernel(x):
    y=2*x+5
    return (y>=0)*(y<=255)*y+(y>255)*255
