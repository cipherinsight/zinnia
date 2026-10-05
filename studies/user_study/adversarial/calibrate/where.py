def kernel(x):
    y=2*x+5
    return np.where(y < 0, 0, np.where(y > 255, 255, y))
