def distance(a,b):
    return sum((p-q)**2 for p,q in zip(a,b))

def kernel(x, y):
    return np.array([[distance(a,b) for b in y] for a in x])
