def kernel(x):
    b,h,w,c = x.shape
    blocks = x.reshape(b,h//2,2,w//2,2,c)
    return np.einsum('bipjqc->bijc', blocks)
