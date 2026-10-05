def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        for i in range(1,a.shape[0]-1):
            for j in range(1,a.shape[1]-1):
                value=4*a[i,j]
                for di,dj in ((-1,0),(1,0),(0,-1),(0,1)):
                    value+=a[i+di,j+dj]
                b[i,j]=value//8
        a=b
    return a
