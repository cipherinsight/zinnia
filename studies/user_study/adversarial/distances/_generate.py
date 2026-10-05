import json
from pathlib import Path

ROOT = Path(__file__).parent
CASES = {
'broadcast_square': ('Pairwise singleton-axis broadcasting and sum.', '''def kernel(x, y):
    delta = x[:, None, :] - y[None, :, :]
    return np.sum(delta ** 2, axis=-1)
'''),
'expand_dims_multiply': ('Explicit expand_dims with multiplication and method reduction.', '''def kernel(x, y):
    delta = np.expand_dims(x, 1) - np.expand_dims(y, 0)
    return (delta * delta).sum(axis=2)
'''),
'reshape_broadcast': ('Shape unpacking and reshape to expose pair axes.', '''def kernel(x, y):
    n, d = x.shape
    m, _ = y.shape
    delta = x.reshape(n, 1, d) - y.reshape(1, m, d)
    return np.sum(np.square(delta), axis=2)
'''),
'gram_matmul': ('Squared norms plus matrix multiplication identity.', '''def kernel(x, y):
    xx = np.sum(x * x, axis=1, keepdims=True)
    yy = np.sum(y * y, axis=1, keepdims=True)
    return xx + yy.T - 2 * (x @ y.T)
'''),
'gram_dot': ('Dot-product Gram matrix with vector norms.', '''def kernel(x, y):
    xx = np.sum(x ** 2, axis=1)
    yy = np.sum(y ** 2, axis=1)
    return xx[:, None] + yy[None, :] - 2 * np.dot(x, np.transpose(y))
'''),
'gram_einsum': ('Einstein summation for norms and rectangular Gram matrix.', '''def kernel(x, y):
    xx = np.einsum('id,id->i', x, x)
    yy = np.einsum('jd,jd->j', y, y)
    xy = np.einsum('id,jd->ij', x, y)
    return xx[:, None] + yy[None, :] - 2 * xy
'''),
'delta_einsum': ('Contract an explicitly broadcast difference tensor.', '''def kernel(x, y):
    delta = x[:, None, :] - y[None, :, :]
    return np.einsum('ijd,ijd->ij', delta, delta)
'''),
'gram_tensordot': ('Contract the feature axes with tensordot.', '''def kernel(x, y):
    return np.sum(x*x, axis=1)[:,None] + np.sum(y*y, axis=1)[None,:] - 2*np.tensordot(x,y,axes=([1],[1]))
'''),
'gram_inner': ('Use inner to compute all row pair dot products.', '''def kernel(x, y):
    return np.sum(x*x, axis=1)[:,None] + np.sum(y*y, axis=1)[None,:] - 2*np.inner(x,y)
'''),
'loops_indexed': ('Triple indexed loop with integer output allocation.', '''def kernel(x, y):
    n, d = x.shape
    m = y.shape[0]
    result = np.zeros((n,m), dtype=int)
    for i in range(n):
        for j in range(m):
            value = 0
            for k in range(d):
                diff = x[i,k] - y[j,k]
                value += diff * diff
            result[i,j] = value
    return result
'''),
'loops_enumerate_zip': ('Enumerate rows and zip coordinates, assign into output.', '''def kernel(x, y):
    result = np.zeros((len(x),len(y)), dtype=int)
    for i, a in enumerate(x):
        for j, b in enumerate(y):
            value = 0
            for p, q in zip(a,b):
                value += (p-q)**2
            result[i,j] = value
    return result
'''),
'nested_comprehension': ('Nested list comprehensions and scalar generator sum.', '''def kernel(x, y):
    return np.array([[sum((p-q)**2 for p,q in zip(a,b)) for b in y] for a in x])
'''),
'row_vectorized_append': ('Append each vectorized row of pairwise distances.', '''def kernel(x, y):
    rows = []
    for a in x:
        delta = y-a
        rows.append(np.sum(delta*delta,axis=1))
    return np.stack(rows)
'''),
'row_list_stack': ('List comprehension with array reductions and stack.', '''def kernel(x, y):
    return np.stack([np.sum((y-a)**2, axis=1) for a in x], axis=0)
'''),
'feature_accumulation': ('Accumulate one broadcast feature contribution at a time.', '''def kernel(x, y):
    result = np.zeros((x.shape[0], y.shape[0]), dtype=int)
    for k in range(x.shape[1]):
        delta = x[:,k,None] - y[None,:,k]
        result += delta*delta
    return result
'''),
'feature_zip_transpose': ('Zip transposed feature columns and use subtract.outer.', '''def kernel(x, y):
    result = np.zeros((len(x),len(y)), dtype=int)
    for a,b in zip(x.T,y.T):
        delta = np.subtract.outer(a,b)
        result += np.square(delta)
    return result
'''),
'local_helper': ('A local scalar row distance helper.', '''def kernel(x, y):
    def distance(a,b):
        delta = a-b
        return np.sum(delta*delta)
    return np.array([[distance(a,b) for b in y] for a in x])
'''),
'global_helper': ('An ordinary module-level helper for each pair.', '''def distance(a,b):
    return sum((p-q)**2 for p,q in zip(a,b))

def kernel(x, y):
    return np.array([[distance(a,b) for b in y] for a in x])
'''),
'row_dot_pairs': ('Pairwise loop using dot of difference vector.', '''def kernel(x, y):
    result = np.zeros((len(x),len(y)), dtype=int)
    for i in range(len(x)):
        for j in range(len(y)):
            delta = x[i]-y[j]
            result[i,j] = np.dot(delta,delta)
    return result
'''),
'repeat_tile': ('Explicitly repeat and tile rows before subtraction.', '''def kernel(x, y):
    n,d = x.shape
    m = y.shape[0]
    a = np.repeat(x,m,axis=0)
    b = np.tile(y,(n,1))
    return np.sum((a-b)**2,axis=1).reshape(n,m)
'''),
'ndindex_pairs': ('Use ndindex to enumerate all output coordinates.', '''def kernel(x, y):
    result = np.zeros((x.shape[0],y.shape[0]),dtype=int)
    for i,j in np.ndindex(result.shape):
        result[i,j] = np.sum(np.square(x[i]-y[j]))
    return result
'''),
'ufunc_reduce': ('Sum squared differences with the add ufunc reduction.', '''def kernel(x, y):
    delta = np.subtract(x[:,None,:],y[None,:,:])
    return np.add.reduce(np.multiply(delta,delta),axis=2)
'''),
'moveaxis_sum': ('Move feature axis first and use built-in sum over arrays.', '''def kernel(x, y):
    delta = x[:,None,:] - y[None,:,:]
    return sum(np.moveaxis(delta*delta,2,0))
'''),
'output_buffer': ('Use ufunc output buffers to avoid allocating squared tensor.', '''def kernel(x, y):
    delta = x[:,None,:] - y[None,:,:]
    np.square(delta,out=delta)
    result = np.zeros((len(x),len(y)),dtype=int)
    np.sum(delta,axis=2,out=result)
    return result
'''),
'while_pairs': ('Nested while loops with vectorized feature reduction.', '''def kernel(x, y):
    result = np.zeros((len(x),len(y)),dtype=int)
    i = 0
    while i < len(x):
        j = 0
        while j < len(y):
            result[i,j] = np.sum((x[i]-y[j])**2)
            j += 1
        i += 1
    return result
'''),
}
items=[]
for name,(intent,source) in CASES.items():
    path=ROOT/(name+'.py')
    path.write_text(source)
    items.append({'name':name,'path':str(path),'intent':intent})
(ROOT/'inventory.json').write_text(json.dumps(items,indent=2)+'\n')
