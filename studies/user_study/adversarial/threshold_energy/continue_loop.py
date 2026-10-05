def kernel(x, floor):
    result = []
    for row in x:
        total = 0
        for c, v in enumerate(row):
            if v <= floor[c]:
                continue
            total += v * v
        result.append(total)
    return np.array(result)
