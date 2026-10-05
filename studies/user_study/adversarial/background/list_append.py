def kernel(x):
    rows = []
    low = x.min(axis=0)
    for row in x:
        rows.append(row - low)
    return np.array(rows)
