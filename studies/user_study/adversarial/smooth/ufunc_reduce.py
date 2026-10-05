def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        b[1:-1,1:-1]=np.add.reduce([4*a[1:-1,1:-1],a[:-2,1:-1],a[2:,1:-1],a[1:-1,:-2],a[1:-1,2:]])//8
        a=b
    return a
