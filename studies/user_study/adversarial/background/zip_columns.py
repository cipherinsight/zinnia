def kernel(x):
    columns = list(zip(*x))
    low = np.array([min(column) for column in columns])
    return x - low
