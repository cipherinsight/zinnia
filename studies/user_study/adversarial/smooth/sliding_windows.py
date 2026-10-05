def kernel(x):
    a=x.copy()
    if min(a.shape)<3:
        return a
    for _ in range(3):
        w=np.lib.stride_tricks.sliding_window_view(a,(3,3))
        b=a.copy()
        b[1:-1,1:-1]=(4*w[:,:,1,1]+w[:,:,0,1]+w[:,:,2,1]+w[:,:,1,0]+w[:,:,1,2])//8
        a=b
    return a
