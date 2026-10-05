def kernel(x):
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        low = 4096
        for item in x[:, c]:
            low = min(low, item)
        out[:, c] = x[:, c] - low
    return out
