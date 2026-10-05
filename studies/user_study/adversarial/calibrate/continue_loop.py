def kernel(x):
    out=np.zeros_like(x)
    for i in range(len(x)):
        y=2*x[i]+5
        if y < 0:
            continue
        if y > 255:
            out[i]=255
            continue
        out[i]=y
    return out
