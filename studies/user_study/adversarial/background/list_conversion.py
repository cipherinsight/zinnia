def kernel(x):
    rows = x.tolist()
    low = [min(row[c] for row in rows) for c in range(x.shape[1])]
    return [[row[c] - low[c] for c in range(x.shape[1])] for row in rows]
