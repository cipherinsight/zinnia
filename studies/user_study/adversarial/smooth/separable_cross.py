def kernel(x):
    a=x.copy()
    for _ in range(3):
        horizontal=a[1:-1,:-2]+2*a[1:-1,1:-1]+a[1:-1,2:]
        vertical=a[:-2,1:-1]+2*a[1:-1,1:-1]+a[2:,1:-1]
        b=a.copy()
        b[1:-1,1:-1]=(horizontal+vertical)//8
        a=b
    return a
