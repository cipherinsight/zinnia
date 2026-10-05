def kernel(x, y):
    result = np.zeros((len(x),len(y)), dtype=int)
    for a,b in zip(x.T,y.T):
        delta = np.subtract.outer(a,b)
        result += np.square(delta)
    return result
