def kernel(x):
    h,w=x.shape
    a=x.flatten()
    for _ in range(3):
        b=a.copy()
        for i in range(1,h-1):
            for j in range(1,w-1):
                k=i*w+j
                b[k]=(4*a[k]+a[k-w]+a[k+w]+a[k-1]+a[k+1])//8
        a=b
    return a.reshape(h,w)
