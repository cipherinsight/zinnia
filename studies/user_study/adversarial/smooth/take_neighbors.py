def kernel(x):
    a=x.copy()
    h,w=x.shape
    if h<3 or w<3:
        return a
    rows=np.arange(1,h-1)
    cols=np.arange(1,w-1)
    for _ in range(3):
        c=np.take(np.take(a,rows,axis=0),cols,axis=1)
        up=np.take(np.take(a,rows-1,axis=0),cols,axis=1)
        down=np.take(np.take(a,rows+1,axis=0),cols,axis=1)
        left=np.take(np.take(a,rows,axis=0),cols-1,axis=1)
        right=np.take(np.take(a,rows,axis=0),cols+1,axis=1)
        b=a.copy()
        b[1:-1,1:-1]=(4*c+up+down+left+right)//8
        a=b
    return a
