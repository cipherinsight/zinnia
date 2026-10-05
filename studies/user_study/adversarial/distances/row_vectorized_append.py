def kernel(x, y):
    rows = []
    for a in x:
        delta = y-a
        rows.append(np.sum(delta*delta,axis=1))
    return np.stack(rows)
