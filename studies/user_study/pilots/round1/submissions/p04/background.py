def kernel(x):
    return x - x.min(axis=0, keepdims=True)
