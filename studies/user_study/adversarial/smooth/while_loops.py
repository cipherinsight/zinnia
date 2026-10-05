def kernel(x):
    a=x.copy()
    r=0
    while r<3:
        b=a.copy()
        i=1
        while i<a.shape[0]-1:
            j=1
            while j<a.shape[1]-1:
                b[i,j]=(4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8
                j+=1
            i+=1
        a=b
        r+=1
    return a
