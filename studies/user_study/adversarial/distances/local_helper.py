def kernel(x, y):
    def distance(a,b):
        delta = a-b
        return np.sum(delta*delta)
    return np.array([[distance(a,b) for b in y] for a in x])
