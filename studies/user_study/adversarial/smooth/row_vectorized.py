def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        for i in range(1,a.shape[0]-1):
            b[i,1:-1]=(4*a[i,1:-1]+a[i-1,1:-1]+a[i+1,1:-1]+a[i,:-2]+a[i,2:])//8
        a=b
    return a
