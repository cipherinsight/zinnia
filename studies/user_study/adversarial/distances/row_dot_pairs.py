def kernel(x, y):
    result = np.zeros((len(x),len(y)), dtype=int)
    for i in range(len(x)):
        for j in range(len(y)):
            delta = x[i]-y[j]
            result[i,j] = np.dot(delta,delta)
    return result
