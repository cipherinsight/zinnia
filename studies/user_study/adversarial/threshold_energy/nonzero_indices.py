def kernel(x, floor):
    result = []
    for row in x:
        total = 0
        for c in np.nonzero(row > floor)[0]:
            total += row[c] ** 2
        result.append(total)
    return np.asarray(result)
