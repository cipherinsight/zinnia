def kernel(x, y):
    result = np.zeros((len(x),len(y)),dtype=int)
    i = 0
    while i < len(x):
        j = 0
        while j < len(y):
            result[i,j] = np.sum((x[i]-y[j])**2)
            j += 1
        i += 1
    return result
