def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        for i,j in np.ndindex(a.shape):
            if i>0 and j>0 and i<a.shape[0]-1 and j<a.shape[1]-1:
                b[i,j]=(4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8
        a=b
    return a
