def kernel(x):
    b,h,w,c = x.shape
    return x.reshape(b,h//2,2,w//2,2,c).sum(axis=4).sum(axis=2)
