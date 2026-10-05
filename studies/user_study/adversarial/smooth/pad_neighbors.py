def kernel(x):
    a=x.copy()
    for _ in range(3):
        p=np.pad(a,1,mode='edge')
        b=(4*a+p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])//8
        b[0,:]=a[0,:]
        b[-1,:]=a[-1,:]
        b[:,0]=a[:,0]
        b[:,-1]=a[:,-1]
        a=b
    return a
