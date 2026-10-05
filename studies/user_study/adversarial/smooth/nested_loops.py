def kernel(x):
    a=x.copy()
    for r in range(3):
        b=a.copy()
        for i in range(1,len(a)-1):
            for j in range(1,len(a[0])-1):
                b[i,j]=(4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8
        a=b
    return a
