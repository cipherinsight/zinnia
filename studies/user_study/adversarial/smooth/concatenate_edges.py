def kernel(x):
    a=x.copy()
    if min(a.shape)<3:
        return a
    for _ in range(3):
        core=(4*a[1:-1,1:-1]+a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:])//8
        middle=np.concatenate((a[1:-1,:1],core,a[1:-1,-1:]),axis=1)
        a=np.concatenate((a[:1,:],middle,a[-1:,:]),axis=0)
    return a
