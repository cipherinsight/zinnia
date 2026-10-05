def kernel(x):
    out=[]
    for v in x:
        out.append(min(255,max(0,2*v+5)))
    return np.asarray(out)
