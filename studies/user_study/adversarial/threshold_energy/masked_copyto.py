def kernel(x, floor):
    squares = x * x
    np.copyto(squares, 0, where=x <= floor)
    return squares.sum(axis=1)
