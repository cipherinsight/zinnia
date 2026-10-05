def kernel(x):
    out = np.zeros_like(x)
    c = 0
    while c < x.shape[1]:
        low = x[0, c]
        t = 1
        while t < x.shape[0]:
            low = min(low, x[t, c])
            t += 1
        t = 0
        while t < x.shape[0]:
            out[t, c] = x[t, c] - low
            t += 1
        c += 1
    return out
