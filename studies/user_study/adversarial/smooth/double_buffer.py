def kernel(x):
    a=x.copy()
    b=x.copy()
    for _ in range(3):
        for i in range(1,x.shape[0]-1):
            for j in range(1,x.shape[1]-1):
                b[i,j]=(4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8
        a,b=b,a
    return a
