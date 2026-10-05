def kernel(x, y):
    return np.array([[sum((p-q)**2 for p,q in zip(a,b)) for b in y] for a in x])
