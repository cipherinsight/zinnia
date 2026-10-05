from pathlib import Path
import json
root=Path(__file__).parent
variants={
'four_slices':('Add the four strided pixel grids directly.', '''def kernel(x):
    return x[:, ::2, ::2, :] + x[:, 1::2, ::2, :] + x[:, ::2, 1::2, :] + x[:, 1::2, 1::2, :]
'''),
'loops_scalar':('Write each batch, output row, output column and channel using scalar loops.', '''def kernel(x):
    b, h, w, c = x.shape
    out = np.zeros((b, h//2, w//2, c), dtype=int)
    for n in range(b):
        for i in range(h//2):
            for j in range(w//2):
                for k in range(c):
                    out[n,i,j,k] = x[n,2*i,2*j,k] + x[n,2*i+1,2*j,k] + x[n,2*i,2*j+1,k] + x[n,2*i+1,2*j+1,k]
    return out
'''),
'patch_sum':('Loop over spatial blocks and reduce each 2x2 patch over spatial axes.', '''def kernel(x):
    b,h,w,c = x.shape
    out = np.zeros((b,h//2,w//2,c), dtype=int)
    for i in range(h//2):
        for j in range(w//2):
            out[:,i,j,:] = np.sum(x[:,2*i:2*i+2,2*j:2*j+2,:], axis=(1,2))
    return out
'''),
'nested_list':('Build the result using nested list comprehensions and scalar sums.', '''def kernel(x):
    b,h,w,c = x.shape
    return np.array([[[[sum(x[n,2*i+di,2*j+dj,k] for di in range(2) for dj in range(2)) for k in range(c)] for j in range(w//2)] for i in range(h//2)] for n in range(b)])
'''),
'append_blocks':('Append block sums to ordinary Python lists before converting to an array.', '''def kernel(x):
    batches = []
    for batch in x:
        rows = []
        for i in range(0, batch.shape[0], 2):
            row = []
            for j in range(0, batch.shape[1], 2):
                row.append(batch[i:i+2,j:j+2,:].sum(axis=(0,1)))
            rows.append(row)
        batches.append(rows)
    return np.array(batches)
'''),
'enumerate_batches':('Enumerate input batches and assign vector block sums.', '''def kernel(x):
    b,h,w,c = x.shape
    out = np.zeros((b,h//2,w//2,c), dtype=int)
    for n, batch in enumerate(x):
        for i in range(h//2):
            for j in range(w//2):
                out[n,i,j] = batch[2*i,2*j] + batch[2*i+1,2*j] + batch[2*i,2*j+1] + batch[2*i+1,2*j+1]
    return out
'''),
'zip_rows':('Zip alternating row grids, then pair columns in each row.', '''def kernel(x):
    batches = []
    for batch in x:
        rows = []
        for top, bottom in zip(batch[::2], batch[1::2]):
            row = top + bottom
            rows.append(row[::2] + row[1::2])
        batches.append(rows)
    return np.array(batches)
'''),
'while_blocks':('Use while loops to traverse non-overlapping blocks.', '''def kernel(x):
    b,h,w,c = x.shape
    out = np.zeros((b,h//2,w//2,c), dtype=int)
    i = 0
    while i < h:
        j = 0
        while j < w:
            out[:,i//2,j//2,:] = x[:,i,j,:] + x[:,i+1,j,:] + x[:,i,j+1,:] + x[:,i+1,j+1,:]
            j += 2
        i += 2
    return out
'''),
'local_helper':('Factor the four-grid addition into a local helper.', '''def kernel(x):
    def add_pairs(a):
        return a[:,::2,:,:] + a[:,1::2,:,:]
    rows = add_pairs(x)
    return rows[:,:,::2,:] + rows[:,:,1::2,:]
'''),
'closure_helper':('Capture the input in a local block helper.', '''def kernel(x):
    b,h,w,c = x.shape
    def block(i,j):
        return x[:,i,j,:] + x[:,i+1,j,:] + x[:,i,j+1,:] + x[:,i+1,j+1,:]
    out = np.zeros((b,h//2,w//2,c),dtype=int)
    for i in range(h//2):
        for j in range(w//2):
            out[:,i,j,:] = block(2*i,2*j)
    return out
'''),
'reshape_tuple_sum':('Expose block axes through reshape and sum both axes together.', '''def kernel(x):
    b,h,w,c = x.shape
    return x.reshape(b,h//2,2,w//2,2,c).sum(axis=(2,4))
'''),
'reshape_sequential_sum':('Reduce horizontal then vertical pair axes separately.', '''def kernel(x):
    b,h,w,c = x.shape
    return x.reshape(b,h//2,2,w//2,2,c).sum(axis=4).sum(axis=2)
'''),
'swapaxes_blocks':('Move the vertical pair axis beside the horizontal pair axis and flatten each block.', '''def kernel(x):
    b,h,w,c = x.shape
    blocks = x.reshape(b,h//2,2,w//2,2,c).swapaxes(2,3)
    return blocks.reshape(b,h//2,w//2,4,c).sum(axis=3)
'''),
'channel_first':('Transpose to channel-first layout for pooling then restore NHWC.', '''def kernel(x):
    b,h,w,c = x.shape
    y = x.transpose(0,3,1,2).reshape(b,c,h//2,2,w//2,2)
    return y.sum(axis=(-1,-3)).transpose(0,2,3,1)
'''),
'einsum':('Sum the two block axes using an explicit einsum contraction.', '''def kernel(x):
    b,h,w,c = x.shape
    blocks = x.reshape(b,h//2,2,w//2,2,c)
    return np.einsum('bipjqc->bijc', blocks)
'''),
'ufunc_reduce':('Use NumPy add.reduce over the exposed block axes.', '''def kernel(x):
    b,h,w,c = x.shape
    return np.add.reduce(x.reshape(b,h//2,2,w//2,2,c), axis=(2,4))
'''),
'reduceat':('Perform segmented addition every two rows and columns with add.reduceat.', '''def kernel(x):
    y = np.add.reduceat(x, np.arange(0,x.shape[1],2), axis=1)
    return np.add.reduceat(y, np.arange(0,x.shape[2],2), axis=2)
'''),
'window_view':('Select every other sliding 2x2 window and sum the window axes.', '''def kernel(x):
    windows = np.lib.stride_tricks.sliding_window_view(x, (2,2), axis=(1,2))
    return windows[:,::2,::2,:,:,:].sum(axis=(-1,-2))
'''),
'stack_slices':('Stack the four strided grids along a new axis before summing.', '''def kernel(x):
    parts = [x[:,::2,::2,:], x[:,1::2,::2,:], x[:,::2,1::2,:], x[:,1::2,1::2,:]]
    return np.stack(parts,axis=0).sum(axis=0)
'''),
'python_sum_slices':('Use Python sum over a generator of the four strided grids.', '''def kernel(x):
    return sum(x[:,i::2,j::2,:] for i in range(2) for j in range(2))
'''),
'copy_inplace':('Copy the first pixel grid and accumulate the other three in place.', '''def kernel(x):
    out = x[:,::2,::2,:].copy()
    out += x[:,1::2,::2,:]
    out += x[:,::2,1::2,:]
    out += x[:,1::2,1::2,:]
    return out
'''),
'add_out':('Accumulate paired rows and columns with the NumPy add out argument.', '''def kernel(x):
    y = np.zeros_like(x[:,::2,:,:])
    np.add(x[:,::2,:,:],x[:,1::2,:,:],out=y)
    out = np.zeros_like(y[:,:,::2,:])
    np.add(y[:,:,::2,:],y[:,:,1::2,:],out=out)
    return out
'''),
'split_pairs':('Split each batch into two-row and two-column blocks before reducing.', '''def kernel(x):
    b,h,w,c = x.shape
    return np.array([[ [block.sum(axis=(0,1)) for block in np.split(rows,w//2,axis=1)] for rows in np.split(batch,h//2,axis=0)] for batch in x])
'''),
'ndindex':('Traverse output coordinates with np.ndindex and sum four scalar pixels.', '''def kernel(x):
    b,h,w,c = x.shape
    out = np.zeros((b,h//2,w//2,c),dtype=int)
    for n,i,j,k in np.ndindex(out.shape):
        out[n,i,j,k] = x[n,2*i,2*j,k] + x[n,2*i+1,2*j,k] + x[n,2*i,2*j+1,k] + x[n,2*i+1,2*j+1,k]
    return out
'''),
}
items=[]
for name,(intent,source) in variants.items():
    filename='scripts_'+name+'.py'
    (root/filename).write_text(source)
    items.append({'name':'scripts_'+name,'path':filename,'intent':intent})
(root/'inventory.json').write_text(json.dumps(items,indent=2)+'\n')
