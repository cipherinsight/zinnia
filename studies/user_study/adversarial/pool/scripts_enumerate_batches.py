def kernel(x):
    b,h,w,c = x.shape
    out = np.zeros((b,h//2,w//2,c), dtype=int)
    for n, batch in enumerate(x):
        for i in range(h//2):
            for j in range(w//2):
                out[n,i,j] = batch[2*i,2*j] + batch[2*i+1,2*j] + batch[2*i,2*j+1] + batch[2*i+1,2*j+1]
    return out
