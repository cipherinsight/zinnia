def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        for i,row in enumerate(a):
            for j,value in enumerate(row):
                if 0<i<a.shape[0]-1 and 0<j<a.shape[1]-1:
                    b[i,j]=(4*value+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8
        a=b
    return a
