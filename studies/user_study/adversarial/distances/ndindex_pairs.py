def kernel(x, y):
    result = np.zeros((x.shape[0],y.shape[0]),dtype=int)
    for i,j in np.ndindex(result.shape):
        result[i,j] = np.sum(np.square(x[i]-y[j]))
    return result
