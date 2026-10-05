def kernel(x, floor):
    result = []
    for row in x:
        terms = []
        for v, f in zip(row, floor):
            if v > f:
                terms.append(v * v)
        result.append(sum(terms))
    return np.asarray(result)
