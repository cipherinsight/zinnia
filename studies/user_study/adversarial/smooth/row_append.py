def kernel(x):
    a=x.copy()
    h,w=x.shape
    for _ in range(3):
        rows=[]
        for i in range(h):
            row=[]
            for j in range(w):
                if i==0 or j==0 or i==h-1 or j==w-1:
                    row.append(a[i,j])
                else:
                    row.append((4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8)
            rows.append(row)
        a=np.array(rows)
    return a
