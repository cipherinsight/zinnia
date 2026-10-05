def step(a):
    b=a.copy()
    b[1:-1,1:-1]=(4*a[1:-1,1:-1]+a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:])//8
    return b

def kernel(x):
    return step(step(step(x)))
