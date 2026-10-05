def kernel(x, floor):
    result = []
    for row in x:
        active = np.take(row, np.flatnonzero(row > floor))
        result.append(np.sum(active * active))
    return np.array(result)
