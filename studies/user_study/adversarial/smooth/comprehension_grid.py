def kernel(x):
    a=x.copy()
    h,w=x.shape
    for _ in range(3):
        a=np.array([[a[i,j] if i==0 or j==0 or i==h-1 or j==w-1 else (4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8 for j in range(w)] for i in range(h)])
    return a
