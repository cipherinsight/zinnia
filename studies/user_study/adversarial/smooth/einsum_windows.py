def kernel(x):
    a=x.copy()
    if min(a.shape)<3:
        return a
    weights=np.array([[0,1,0],[1,4,1],[0,1,0]])
    for _ in range(3):
        w=np.lib.stride_tricks.sliding_window_view(a,(3,3))
        b=a.copy()
        b[1:-1,1:-1]=np.einsum('ijab,ab->ij',w,weights)//8
        a=b
    return a
