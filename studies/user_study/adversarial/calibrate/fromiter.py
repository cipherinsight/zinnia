def kernel(x):
    return np.fromiter((min(255,max(0,2*v+5)) for v in x), dtype=np.int64, count=len(x))
