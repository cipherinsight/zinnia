from pathlib import Path
import json,textwrap
root=Path(__file__).parent
items=[]
def add(name,intent,src):
 p=root/(name+'.py');p.write_text(textwrap.dedent(src).strip()+'\n');items.append(dict(name=name,path=str(p),intent=intent))
add('slice_copy','Vectorized five slices with a fresh synchronous copy.', '''
def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        b[1:-1,1:-1]=(4*a[1:-1,1:-1]+a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:])//8
        a=b
    return a
''')
add('double_buffer','Reuse two arrays and swap after each complete round.', '''
def kernel(x):
    a=x.copy()
    b=x.copy()
    for _ in range(3):
        for i in range(1,x.shape[0]-1):
            for j in range(1,x.shape[1]-1):
                b[i,j]=(4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8
        a,b=b,a
    return a
''')
add('nested_loops','Scalar indexed loops and fresh copy each round.', '''
def kernel(x):
    a=x.copy()
    for r in range(3):
        b=a.copy()
        for i in range(1,len(a)-1):
            for j in range(1,len(a[0])-1):
                b[i,j]=(4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8
        a=b
    return a
''')
add('while_loops','Three nested while loops express fixed iteration and interior traversal.', '''
def kernel(x):
    a=x.copy()
    r=0
    while r<3:
        b=a.copy()
        i=1
        while i<a.shape[0]-1:
            j=1
            while j<a.shape[1]-1:
                b[i,j]=(4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8
                j+=1
            i+=1
        a=b
        r+=1
    return a
''')
add('row_vectorized','Iterate rows while vectorizing columns.', '''
def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        for i in range(1,a.shape[0]-1):
            b[i,1:-1]=(4*a[i,1:-1]+a[i-1,1:-1]+a[i+1,1:-1]+a[i,:-2]+a[i,2:])//8
        a=b
    return a
''')
add('column_vectorized','Iterate columns while vectorizing rows.', '''
def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        for j in range(1,a.shape[1]-1):
            b[1:-1,j]=(4*a[1:-1,j]+a[:-2,j]+a[2:,j]+a[1:-1,j-1]+a[1:-1,j+1])//8
        a=b
    return a
''')
add('offset_accumulation','Accumulate a reusable four-neighbor offset stencil.', '''
def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        for i in range(1,a.shape[0]-1):
            for j in range(1,a.shape[1]-1):
                value=4*a[i,j]
                for di,dj in ((-1,0),(1,0),(0,-1),(0,1)):
                    value+=a[i+di,j+dj]
                b[i,j]=value//8
        a=b
    return a
''')
add('ndindex_guard','Traverse all coordinates with ndindex and skip fixed boundaries.', '''
def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        for i,j in np.ndindex(a.shape):
            if i>0 and j>0 and i<a.shape[0]-1 and j<a.shape[1]-1:
                b[i,j]=(4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8
        a=b
    return a
''')
add('flat_indices','Flatten arrays and use row-major neighbor offsets.', '''
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
''')
add('list_grid','Nested Python lists with scalar traversal and array conversion.', '''
def kernel(x):
    a=x.tolist()
    h,w=x.shape
    for _ in range(3):
        b=[row[:] for row in a]
        for i in range(1,h-1):
            for j in range(1,w-1):
                b[i][j]=(4*a[i][j]+a[i-1][j]+a[i+1][j]+a[i][j-1]+a[i][j+1])//8
        a=b
    return np.array(a)
''')
add('comprehension_grid','Whole-grid nested comprehension with boundary conditional.', '''
def kernel(x):
    a=x.copy()
    h,w=x.shape
    for _ in range(3):
        a=np.array([[a[i,j] if i==0 or j==0 or i==h-1 or j==w-1 else (4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8 for j in range(w)] for i in range(h)])
    return a
''')
add('row_append','Build output as appended rows and cells.', '''
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
''')
add('local_helper','Local helper implements one synchronous round.', '''
def kernel(x):
    def step(a):
        b=a.copy()
        b[1:-1,1:-1]=(4*a[1:-1,1:-1]+a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:])//8
        return b
    a=x.copy()
    for _ in range(3):
        a=step(a)
    return a
''')
add('global_helper','Top-level pure helper and explicit three-round composition.', '''
def step(a):
    b=a.copy()
    b[1:-1,1:-1]=(4*a[1:-1,1:-1]+a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:])//8
    return b

def kernel(x):
    return step(step(step(x)))
''')
add('roll_restore','Compute full-grid rolled neighbors then restore boundary.', '''
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
''')
add('pad_neighbors','Edge padding supplies neighbor arrays; only interior is replaced.', '''
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
''')
add('sliding_windows','Use sliding_window_view and a 3x3 cross stencil.', '''
def kernel(x):
    a=x.copy()
    if min(a.shape)<3:
        return a
    for _ in range(3):
        w=np.lib.stride_tricks.sliding_window_view(a,(3,3))
        b=a.copy()
        b[1:-1,1:-1]=(4*w[:,:,1,1]+w[:,:,0,1]+w[:,:,2,1]+w[:,:,1,0]+w[:,:,1,2])//8
        a=b
    return a
''')
add('stack_sum','Stack weighted neighbor planes and sum along leading axis.', '''
def kernel(x):
    a=x.copy()
    for _ in range(3):
        neighbors=np.stack((4*a[1:-1,1:-1],a[:-2,1:-1],a[2:,1:-1],a[1:-1,:-2],a[1:-1,2:]),axis=0)
        b=a.copy()
        b[1:-1,1:-1]=np.sum(neighbors,axis=0)//8
        a=b
    return a
''')
add('ufunc_reduce','Use add.reduce over weighted neighbor stack.', '''
def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        b[1:-1,1:-1]=np.add.reduce([4*a[1:-1,1:-1],a[:-2,1:-1],a[2:,1:-1],a[1:-1,:-2],a[1:-1,2:]])//8
        a=b
    return a
''')
add('einsum_windows','Contract sliding windows with an integer stencil using einsum.', '''
def kernel(x):
    a=x.copy()
    if min(a.shape)<3:
        return a
    weights=np.array([[0,1,0],[1,4,1],[0,1,0]])
    for _ in range(3):
        w=np.lib.stride_tricks.sliding_window_view(a,(3,3))
        b=a.copy()
        b[1:-1,1:-1]=np.einsum('ijab,ab->ij',w,weights)//8
        a=b
    return a
''')
add('separable_cross','Sum horizontal and vertical 1D filters, sharing central weight.', '''
def kernel(x):
    a=x.copy()
    for _ in range(3):
        horizontal=a[1:-1,:-2]+2*a[1:-1,1:-1]+a[1:-1,2:]
        vertical=a[:-2,1:-1]+2*a[1:-1,1:-1]+a[2:,1:-1]
        b=a.copy()
        b[1:-1,1:-1]=(horizontal+vertical)//8
        a=b
    return a
''')
add('take_neighbors','Gather interior neighbor planes with take and index arrays.', '''
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
''')
add('transpose_round','Transpose axes to reuse row-oriented processing.', '''
def kernel(x):
    a=x.T.copy()
    for _ in range(3):
        b=a.copy()
        for i in range(1,a.shape[0]-1):
            b[i,1:-1]=(4*a[i,1:-1]+a[i-1,1:-1]+a[i+1,1:-1]+a[i,:-2]+a[i,2:])//8
        a=b
    return a.T
''')
add('concatenate_edges','Assemble interior and unchanged edges using concatenation.', '''
def kernel(x):
    a=x.copy()
    if min(a.shape)<3:
        return a
    for _ in range(3):
        core=(4*a[1:-1,1:-1]+a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:])//8
        middle=np.concatenate((a[1:-1,:1],core,a[1:-1,-1:]),axis=1)
        a=np.concatenate((a[:1,:],middle,a[-1:,:]),axis=0)
    return a
''')
add('enumerate_rows','Enumerate rows and cells while preserving boundary values.', '''
def kernel(x):
    a=x.copy()
    for _ in range(3):
        b=a.copy()
        for i,row in enumerate(a):
            for j,value in enumerate(row):
                if 0<i<a.shape[0]-1 and 0<j<a.shape[1]-1:
                    b[i,j]=(4*value+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])//8
        a=b
    return a
''')
(root/'inventory.json').write_text(json.dumps(items,indent=2)+'\n')
