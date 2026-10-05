def kernel(x):
    a=x.copy()
    for _ in range(3):
        neighbors=np.stack((4*a[1:-1,1:-1],a[:-2,1:-1],a[2:,1:-1],a[1:-1,:-2],a[1:-1,2:]),axis=0)
        b=a.copy()
        b[1:-1,1:-1]=np.sum(neighbors,axis=0)//8
        a=b
    return a
