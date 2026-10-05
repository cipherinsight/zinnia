def kernel(x):
    b,h,w,c = x.shape
    return np.add.reduce(x.reshape(b,h//2,2,w//2,2,c), axis=(2,4))
