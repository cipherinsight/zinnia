def kernel(x):
    b,h,w,c = x.shape
    y = x.transpose(0,3,1,2).reshape(b,c,h//2,2,w//2,2)
    return y.sum(axis=(-1,-3)).transpose(0,2,3,1)
