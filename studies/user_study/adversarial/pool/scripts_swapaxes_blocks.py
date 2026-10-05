def kernel(x):
    b,h,w,c = x.shape
    blocks = x.reshape(b,h//2,2,w//2,2,c).swapaxes(2,3)
    return blocks.reshape(b,h//2,w//2,4,c).sum(axis=3)
