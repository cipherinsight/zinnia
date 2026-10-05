def kernel(x, floor):
    return sum((np.where(column > f, column * column, 0) for column, f in zip(x.T, floor)), np.zeros(x.shape[0], dtype=int))
