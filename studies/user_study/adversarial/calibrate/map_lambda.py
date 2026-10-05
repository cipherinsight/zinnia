def kernel(x):
    return np.array(list(map(lambda v: min(255,max(0,2*v+5)), x)))
