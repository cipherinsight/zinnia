def kernel(x):
    def value(t, c):
        return x[t, c] - min(x[:, c])
    return np.array([[value(t, c) for c in range(x.shape[1])] for t in range(x.shape[0])])
