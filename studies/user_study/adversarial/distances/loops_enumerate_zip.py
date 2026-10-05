def kernel(x, y):
    result = np.zeros((len(x),len(y)), dtype=int)
    for i, a in enumerate(x):
        for j, b in enumerate(y):
            value = 0
            for p, q in zip(a,b):
                value += (p-q)**2
            result[i,j] = value
    return result
