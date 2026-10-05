def kernel(x, floor):
    result = []
    for row in x:
        active = row[row > floor]
        result.append(np.dot(active, active))
    return np.asarray(result)
