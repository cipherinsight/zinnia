def kernel(x):
    f=np.vectorize(lambda v: min(255,max(0,2*v+5)))
    return f(x)
