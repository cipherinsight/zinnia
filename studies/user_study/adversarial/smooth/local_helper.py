def kernel(x):
    def step(a):
        b=a.copy()
        b[1:-1,1:-1]=(4*a[1:-1,1:-1]+a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:])//8
        return b
    a=x.copy()
    for _ in range(3):
        a=step(a)
    return a
