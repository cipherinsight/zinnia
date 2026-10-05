def kernel(x):
    def correct(column):
        return column - min(column)
    return np.array([correct(column) for column in x.T]).T
