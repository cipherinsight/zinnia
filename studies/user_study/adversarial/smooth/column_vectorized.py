def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        for j in range(1,a.shape[1]-1):
            b[1:-1,j]=(4*a[1:-1,j]+a[:-2,j]+a[2:,j]+a[1:-1,j-1]+a[1:-1,j+1])//8
        a=b
    return a
