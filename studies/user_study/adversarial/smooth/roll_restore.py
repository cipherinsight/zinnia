def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=(4*a+np.roll(a,1,axis=0)+np.roll(a,-1,axis=0)+np.roll(a,1,axis=1)+np.roll(a,-1,axis=1))//8
        b[0,:]=a[0,:]
        b[-1,:]=a[-1,:]
        b[:,0]=a[:,0]
        b[:,-1]=a[:,-1]
        a=b
    return a
