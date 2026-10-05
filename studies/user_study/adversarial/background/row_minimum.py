def kernel(x):
    low = x[0].copy()
    for row in x[1:]:
        low = np.minimum(low, row)
    return x - low
