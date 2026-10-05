def kernel(x):
    return np.column_stack([column - column.min() for column in x.T])
