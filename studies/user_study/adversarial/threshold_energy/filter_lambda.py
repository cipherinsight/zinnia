def kernel(x, floor):
    result = []
    for row in x:
        pairs = filter(lambda pair: pair[0] > pair[1], zip(row, floor))
        result.append(sum(map(lambda pair: pair[0] * pair[0], pairs)))
    return np.array(result)
