def kernel(x):
    return np.array([min(255,max(0,2*v+5)) for v in x])
