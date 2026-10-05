def kernel(x):
    return sum(x[:,i::2,j::2,:] for i in range(2) for j in range(2))
