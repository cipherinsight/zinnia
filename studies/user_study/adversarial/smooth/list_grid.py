def kernel(x):
    a=x.tolist()
    h,w=x.shape
    for _ in range(3):
        b=[row[:] for row in a]
        for i in range(1,h-1):
            for j in range(1,w-1):
                b[i][j]=(4*a[i][j]+a[i-1][j]+a[i+1][j]+a[i][j-1]+a[i][j+1])//8
        a=b
    return np.array(a)
