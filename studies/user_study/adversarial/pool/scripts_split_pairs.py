def kernel(x):
    b,h,w,c = x.shape
    return np.array([[ [block.sum(axis=(0,1)) for block in np.split(rows,w//2,axis=1)] for rows in np.split(batch,h//2,axis=0)] for batch in x])
