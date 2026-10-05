"""Composed array tasks with independent scalar oracles and diverse solutions.

All dimensions are public and positive. Contents are private bounded integers.
The inventory is a compatibility gate, not a restriction on participant solutions.
"""
from dataclasses import dataclass
from textwrap import dedent
import numpy as np
from ..tasks import chained_indices

@dataclass(frozen=True)
class Task:
    name: str
    profiles: tuple
    arguments: tuple
    output_rank: int
    variants: dict

def body(s):
    return dedent(s).strip()

TASKS = [
 Task('channel_energy_v2', ({'B':2,'T':3,'C':2},{'B':1,'T':1,'C':1},{'B':1,'T':4,'C':3}), ('x','gain','bias'), 2, {
 'broadcast': '''a = np.clip(x * gain + bias, 0, 255)
a = a - np.min(a, axis=1, keepdims=True)
return np.sum(a * a, axis=1)''',
 'expanded_axes': '''a = np.minimum(np.maximum(x * gain[None, None, :] + bias[None, None, :], 0), 255)
a = a - a.min(axis=1)[:, None, :]
return (a ** 2).sum(axis=1)''',
 'where': '''a = x * gain + bias
a = np.where(a < 0, 0, np.where(a > 255, 255, a))
a -= a.min(axis=1, keepdims=True)
return np.einsum('btc,btc->bc', a, a)''',
 'column_slices': body('''
 out = np.zeros((x.shape[0], x.shape[2]), dtype=int)
 for b in range(x.shape[0]):
     for c in range(x.shape[2]):
         a = np.clip(x[b, :, c] * gain[c] + bias[c], 0, 255)
         a = a - min(a)
         out[b, c] = sum(a * a)
 return out
 '''),
 'nested_loops': body('''
 B, T, C = x.shape
 out = np.zeros((B, C), dtype=int)
 for b in range(B):
     for c in range(C):
         low = 255
         for t in range(T):
             value = min(255, max(0, x[b,t,c] * gain[c] + bias[c]))
             low = min(low, value)
         for t in range(T):
             value = min(255, max(0, x[b,t,c] * gain[c] + bias[c])) - low
             out[b,c] += value * value
 return out
 '''),
 'comprehension': body('''
 result = []
 for batch in x:
     row = []
     for c in range(x.shape[2]):
         values = [min(255, max(0, v * gain[c] + bias[c])) for v in batch[:, c]]
         low = min(values)
         row.append(sum((v-low)**2 for v in values))
     result.append(row)
 return np.array(result)
 '''),
 }),
 Task('centered_gram_v2', ({'B':2,'T':3,'C':2},{'B':1,'T':1,'C':1},{'B':1,'T':4,'C':3}), ('x',), 3, {
 'einsum': '''z = x.shape[1] * x - x.sum(axis=1, keepdims=True)
return np.einsum('btc,btd->bcd', z, z)''',
 'batched_matmul': '''z = x.shape[1] * x - np.sum(x, axis=1)[:, None, :]
return np.transpose(z, (0, 2, 1)) @ z''',
 'broadcast': '''z = x.shape[1] * x - np.sum(x, axis=1, keepdims=True)
return np.sum(z[:, :, :, None] * z[:, :, None, :], axis=1)''',
 'batch_dot': body('''
 out = []
 for batch in x:
     z = batch.shape[0] * batch - batch.sum(axis=0)
     out.append(np.dot(z.T, z))
 return np.array(out)
 '''),
 'algebraic_identity': '''T = x.shape[1]
s = x.sum(axis=1)
return T*T*np.einsum('btc,btd->bcd', x, x) - T*s[:, :, None]*s[:, None, :]''',
 'nested_loops': body('''
 B, T, C = x.shape
 out = np.zeros((B,C,C), dtype=int)
 for b in range(B):
     for c in range(C):
         for d in range(C):
             sc = 0
             sd = 0
             for t in range(T):
                 sc += x[b,t,c]
                 sd += x[b,t,d]
             for t in range(T):
                 out[b,c,d] += (T*x[b,t,c]-sc)*(T*x[b,t,d]-sd)
 return out
 '''),
 }),
 Task('weighted_distances_v2', ({'B':2,'N':3,'M':2,'D':3},{'B':1,'N':1,'M':1,'D':1},{'B':1,'N':2,'M':4,'D':2}), ('x','y','weight'), 3, {
 'broadcast': '''delta = x[:, :, None, :] - y[:, None, :, :]
return np.sum(delta * delta * weight, axis=-1)''',
 'einsum': '''delta = x[:, :, None, :] - y[:, None, :, :]
return np.einsum('bnmd,d,bnmd->bnm', delta, weight, delta)''',
 'matmul_identity': '''xx = np.sum(x*x*weight, axis=-1)
yy = np.sum(y*y*weight, axis=-1)
return xx[:, :, None] + yy[:, None, :] - 2 * ((x*weight) @ y.transpose(0,2,1))''',
 'per_batch_dot': body('''
 result = []
 for b in range(x.shape[0]):
     a = x[b]
     other = y[b]
     result.append((a*a*weight).sum(axis=1)[:,None] + (other*other*weight).sum(axis=1)[None,:] - 2*np.dot(a*weight, other.T))
 return np.array(result)
 '''),
 'pair_slices': body('''
 out = np.zeros((x.shape[0],x.shape[1],y.shape[1]), dtype=int)
 for b in range(x.shape[0]):
     for i in range(x.shape[1]):
         for j in range(y.shape[1]):
             delta = x[b,i] - y[b,j]
             out[b,i,j] = np.sum(weight * delta ** 2)
 return out
 '''),
 'nested_loops': body('''
 B, N, D = x.shape
 M = y.shape[1]
 out = np.zeros((B,N,M), dtype=int)
 for b in range(B):
     for i in range(N):
         for j in range(M):
             for d in range(D):
                 delta = x[b,i,d]-y[b,j,d]
                 out[b,i,j] += weight[d]*delta*delta
 return out
 '''),
 }),
 Task('convolution_v2', ({'B':1,'H':4,'W':5,'C':2,'K':2,'U':2,'V':3},{'B':1,'H':1,'W':1,'C':1,'K':1,'U':1,'V':1},{'B':2,'H':3,'W':4,'C':3,'K':2,'U':3,'V':2}), ('x','filters'), 4, {
 'patch_einsum': body('''
 B,H,W,C = x.shape
 K,U,V,C = filters.shape
 out = np.zeros((B,H-U+1,W-V+1,K), dtype=int)
 for i in range(H-U+1):
     for j in range(W-V+1):
         out[:,i,j,:] = np.einsum('buvc,kuvc->bk', x[:,i:i+U,j:j+V,:], filters)
 return out
 '''),
 'offset_tensordot': body('''
 B,H,W,C = x.shape
 K,U,V,C = filters.shape
 out = np.zeros((B,H-U+1,W-V+1,K), dtype=int)
 for u in range(U):
     for v in range(V):
         out += np.tensordot(x[:,u:u+H-U+1,v:v+W-V+1,:], filters[:,u,v,:], axes=([-1],[-1]))
 return out
 '''),
 'patch_broadcast': body('''
 B,H,W,C = x.shape
 K,U,V,C = filters.shape
 out = np.zeros((B,H-U+1,W-V+1,K), dtype=int)
 for i in range(H-U+1):
     for j in range(W-V+1):
         patch = x[:,i:i+U,j:j+V,:]
         out[:,i,j,:] = np.sum(patch[:,None,:,:,:]*filters[None,:,:,:,:], axis=(2,3,4))
 return out
 '''),
 'flatten_dot': body('''
 B,H,W,C = x.shape
 K,U,V,C = filters.shape
 out = np.zeros((B,H-U+1,W-V+1,K), dtype=int)
 f = filters.reshape(K,-1)
 for i in range(H-U+1):
     for j in range(W-V+1):
         patch = x[:,i:i+U,j:j+V,:].reshape(B,-1)
         out[:,i,j,:] = patch @ f.T
 return out
 '''),
 'nested_loops': body('''
 B,H,W,C = x.shape
 K,U,V,C = filters.shape
 out = np.zeros((B,H-U+1,W-V+1,K), dtype=int)
 for b in range(B):
     for i in range(H-U+1):
         for j in range(W-V+1):
             for k in range(K):
                 for u in range(U):
                     for v in range(V):
                         for c in range(C):
                             out[b,i,j,k] += x[b,i+u,j+v,c]*filters[k,u,v,c]
 return out
 '''),
 'patch_sum': body('''
 B,H,W,C = x.shape
 K,U,V,C = filters.shape
 out = np.zeros((B,H-U+1,W-V+1,K), dtype=int)
 for b in range(B):
     for i in range(H-U+1):
         for j in range(W-V+1):
             for k in range(K):
                 out[b,i,j,k] = np.sum(x[b,i:i+U,j:j+V,:] * filters[k])
 return out
 '''),
 }),
 Task('block_covariance_v2', ({'B':1,'H':4,'W':6,'C':2},{'B':1,'H':2,'W':2,'C':1},{'B':2,'H':2,'W':4,'C':3}), ('x',), 5, {
 'reshape_einsum': '''B,H,W,C = x.shape
a = x.reshape(B,H//2,2,W//2,2,C).transpose(0,1,3,2,4,5).reshape(B,H//2,W//2,4,C)
z = 4*a - a.sum(axis=3,keepdims=True)
return np.einsum('bhwpc,bhwpd->bhwcd',z,z)''',
 'stack_matmul': '''a = np.stack([x[:,0::2,0::2,:],x[:,0::2,1::2,:],x[:,1::2,0::2,:],x[:,1::2,1::2,:]],axis=3)
z = 4*a - a.sum(axis=3,keepdims=True)
return z.transpose(0,1,2,4,3) @ z''',
 'broadcast': '''a = np.stack([x[:,0::2,0::2,:],x[:,0::2,1::2,:],x[:,1::2,0::2,:],x[:,1::2,1::2,:]],axis=3)
z = 4*a - a.sum(axis=3,keepdims=True)
return np.sum(z[:,:,:,:, :,None]*z[:,:,:,:,None,:],axis=3)''',
 'block_dot': body('''
 B,H,W,C = x.shape
 out = np.zeros((B,H//2,W//2,C,C),dtype=int)
 for b in range(B):
     for i in range(H//2):
         for j in range(W//2):
             a = x[b,2*i:2*i+2,2*j:2*j+2,:].reshape(4,C)
             z = 4*a-a.sum(axis=0)
             out[b,i,j,:,:] = np.dot(z.T,z)
 return out
 '''),
 'nested_loops': body('''
 B,H,W,C = x.shape
 out = np.zeros((B,H//2,W//2,C,C),dtype=int)
 for b in range(B):
     for i in range(H//2):
         for j in range(W//2):
             for c in range(C):
                 for d in range(C):
                     sc = 0
                     sd = 0
                     for u in range(2):
                         for v in range(2):
                             sc += x[b,2*i+u,2*j+v,c]
                             sd += x[b,2*i+u,2*j+v,d]
                     for u in range(2):
                         for v in range(2):
                             out[b,i,j,c,d] += (4*x[b,2*i+u,2*j+v,c]-sc)*(4*x[b,2*i+u,2*j+v,d]-sd)
 return out
 '''),
 'algebraic_identity': '''a = np.stack([x[:,0::2,0::2,:],x[:,0::2,1::2,:],x[:,1::2,0::2,:],x[:,1::2,1::2,:]],axis=3)
s = a.sum(axis=3)
return 16*np.einsum('bhwpc,bhwpd->bhwcd',a,a)-4*s[:,:,:,:,None]*s[:,:,:,None,:]''',
 }),
 Task('masked_diffusion_v2', ({'B':1,'H':4,'W':5,'C':2},{'B':1,'H':1,'W':2,'C':1},{'B':2,'H':3,'W':4,'C':3}), ('x','mask','mix'), 4, {
 'slices_matmul': body('''
 a = x.copy()
 for step in range(3):
     out = a.copy()
     local = (4*a[:,1:-1,1:-1,:]+a[:,:-2,1:-1,:]+a[:,2:,1:-1,:]+a[:,1:-1,:-2,:]+a[:,1:-1,2:,:])//8
     value = np.clip(local @ mix,0,255)
     out[:,1:-1,1:-1,:] = np.where(mask[None,1:-1,1:-1,None] != 0,value,a[:,1:-1,1:-1,:])
     a = out
 return a
 '''),
 'slices_einsum': body('''
 a = np.copy(x)
 for step in range(3):
     out = np.copy(a)
     local = (4*a[:,1:-1,1:-1,:]+a[:,:-2,1:-1,:]+a[:,2:,1:-1,:]+a[:,1:-1,:-2,:]+a[:,1:-1,2:,:])//8
     value = np.clip(np.einsum('bhwc,cd->bhwd',local,mix),0,255)
     out[:,1:-1,1:-1,:] = np.where(mask[None,1:-1,1:-1,None] != 0,value,a[:,1:-1,1:-1,:])
     a = out
 return a
 '''),
 'cell_dot': body('''
 a = x.copy()
 for step in range(3):
     out = a.copy()
     for b in range(a.shape[0]):
         for i in range(1,a.shape[1]-1):
             for j in range(1,a.shape[2]-1):
                 if mask[i,j] != 0:
                     local = (4*a[b,i,j]+a[b,i-1,j]+a[b,i+1,j]+a[b,i,j-1]+a[b,i,j+1])//8
                     out[b,i,j] = np.clip(np.dot(local,mix),0,255)
     a = out
 return a
 '''),
 'nested_loops': body('''
 B,H,W,C = x.shape
 a = x.copy()
 for step in range(3):
     out = a.copy()
     for b in range(B):
         for i in range(1,H-1):
             for j in range(1,W-1):
                 if mask[i,j] != 0:
                     for d in range(C):
                         total = 0
                         for c in range(C):
                             local = (4*a[b,i,j,c]+a[b,i-1,j,c]+a[b,i+1,j,c]+a[b,i,j-1,c]+a[b,i,j+1,c])//8
                             total += local*mix[c,d]
                         out[b,i,j,d] = min(255,max(0,total))
     a = out
 return a
 '''),
 }),
]

# Include syntactic choices participants actually made in the first pilot.
for task in TASKS:
    task.variants['chained_indexing'] = chained_indices(task.variants['nested_loops'])
    task.variants['default_zeros'] = task.variants['nested_loops'].replace(', dtype=int', '').replace(',dtype=int', '')
# Further natural implementations: function aliases, explicit buffers, and
# shape-bounded while loops. None are removed when a compiler gap is found.
by_name = {t.name: t for t in TASKS}
by_name['centered_gram_v2'].variants['matmul_function'] = "z = x.shape[1]*x - x.sum(axis=1,keepdims=True)\nreturn np.matmul(z.transpose(0,2,1), z)"
by_name['weighted_distances_v2'].variants['comprehension'] = "return np.array([[[sum(weight[d]*(x[b,i,d]-y[b,j,d])**2 for d in range(x.shape[2])) for j in range(y.shape[1])] for i in range(x.shape[1])] for b in range(x.shape[0])])"
by_name['centered_gram_v2'].variants['tensordot_per_batch'] = "z = x.shape[1]*x-x.sum(axis=1,keepdims=True)\nreturn np.array([np.tensordot(batch,batch,axes=([0],[0])) for batch in z])"
by_name['channel_energy_v2'].variants['negative_axis'] = "a = (x*gain+bias).clip(0,255)\na -= a.min(axis=-2,keepdims=True)\nreturn (a*a).sum(axis=-2)"
by_name['channel_energy_v2'].variants['mask_assignment'] = "a = x*gain+bias\na[a<0]=0\na[a>255]=255\na -= a.min(axis=1,keepdims=True)\nreturn (a*a).sum(axis=1)"
by_name['masked_diffusion_v2'].variants['read_copy_write_original'] = by_name['masked_diffusion_v2'].variants['nested_loops'].replace('out = a.copy()', 'old = a.copy()\n    out = a').replace('4*a[','4*old[').replace('+a[','+old[')
by_name['masked_diffusion_v2'].variants['buffer_assignment'] = by_name['masked_diffusion_v2'].variants['slices_einsum'].replace('a = out','a[:, :, :, :] = out')
by_name['convolution_v2'].variants['offset_matmul'] = by_name['convolution_v2'].variants['offset_tensordot'].replace("np.tensordot(x[:,u:u+H-U+1,v:v+W-V+1,:], filters[:,u,v,:], axes=([-1],[-1]))", "x[:,u:u+H-U+1,v:v+W-V+1,:] @ filters[:,u,v,:].T")
# A while-loop implementation of every scalar algorithm exercises symbolic
# bounds independently of for/range lowering.
import ast
class WhileLoops(ast.NodeTransformer):
    def visit_For(self,node):
        self.generic_visit(node)
        if not isinstance(node.iter,ast.Call) or not isinstance(node.iter.func,ast.Name) or node.iter.func.id!='range':return node
        args=node.iter.args
        start=ast.Constant(0) if len(args)==1 else args[0]
        end=args[-1]
        init=ast.Assign(targets=[node.target],value=start)
        var=ast.Name(id=node.target.id,ctx=ast.Load())
        loop=ast.While(test=ast.Compare(left=var,ops=[ast.Lt()],comparators=[end]),body=node.body+[ast.AugAssign(target=node.target,op=ast.Add(),value=ast.Constant(1))],orelse=[])
        return [init,loop]
for task in TASKS:
    tree=WhileLoops().visit(ast.parse(task.variants['nested_loops']))
    task.variants['while_loops']=ast.unparse(ast.fix_missing_locations(tree))

# No-update dimensions are handled by the same ordinary empty-slice semantics.

def signatures(task,p):
    n=task.name
    if n=='channel_energy_v2':
        return {'x':(p['B'],p['T'],p['C']),'gain':(p['C'],),'bias':(p['C'],)}, (p['B'],p['C'])
    if n=='centered_gram_v2':
        return {'x':(p['B'],p['T'],p['C'])}, (p['B'],p['C'],p['C'])
    if n=='weighted_distances_v2':
        return {'x':(p['B'],p['N'],p['D']),'y':(p['B'],p['M'],p['D']),'weight':(p['D'],)}, (p['B'],p['N'],p['M'])
    if n=='convolution_v2':
        return {'x':(p['B'],p['H'],p['W'],p['C']),'filters':(p['K'],p['U'],p['V'],p['C'])}, (p['B'],p['H']-p['U']+1,p['W']-p['V']+1,p['K'])
    if n=='block_covariance_v2':
        return {'x':(p['B'],p['H'],p['W'],p['C'])}, (p['B'],p['H']//2,p['W']//2,p['C'],p['C'])
    return {'x':(p['B'],p['H'],p['W'],p['C']),'mask':(p['H'],p['W']),'mix':(p['C'],p['C'])}, (p['B'],p['H'],p['W'],p['C'])

def source(task,variant,profile):
    shapes,_=signatures(task,profile)
    args=', '.join(f'{n}: Integer[Array, "{" ".join(map(str,s))}"]' for n,s in shapes.items())
    return 'def kernel('+args+'):\n'+'\n'.join('    '+s for s in task.variants[variant].splitlines())+'\n'

def fixtures(task,profile):
    shapes,_=signatures(task,profile)
    rng=np.random.default_rng(20260922)
    result=[]
    for mode in range(8):
        values=[]
        for name,shape in shapes.items():
            if mode==0: a=np.zeros(shape,dtype=np.int64)
            elif mode==1: a=np.full(shape,3,dtype=np.int64)
            elif mode==2: a=np.arange(np.prod(shape),dtype=np.int64).reshape(shape)%17-8
            elif mode==3: a=np.full(shape,-3,dtype=np.int64)
            elif mode==4: a=np.resize(np.array([-16,16,0,1,-1],dtype=np.int64),shape)
            else: a=rng.integers(-16,17,size=shape,dtype=np.int64)
            if name in ('weight','gain'): a=np.abs(a)%5
            if task.name=='channel_energy_v2' and mode==4:
                if name=='x':a=np.resize(np.array([-64,64,0,63],dtype=np.int64),shape)
                elif name=='gain':a=np.full(shape,4,dtype=np.int64)
                elif name=='bias':a=np.full(shape,16,dtype=np.int64)
            if task.name=='masked_diffusion_v2':
                if name=='x': a=np.abs(a)*7
                elif name=='mask': a=np.abs(a)%2
                elif name=='mix': a=a%5-2
                if mode==1 and name=='mix': a=np.eye(shape[0],dtype=np.int64)
                if mode==3 and name=='mask': a=np.ones(shape,dtype=np.int64)
            values.append(a)
        result.append(values)
    return result

def oracle(task,p,values):
    """Deliberately scalar, Python-integer arithmetic; independent of variants."""
    x=values[0];_,shape=signatures(task,p)
    out=np.zeros(shape,dtype=np.int64);n=task.name
    if n=='channel_energy_v2':
        gain,bias=values[1:]
        for b,c in np.ndindex(shape):
            a=[min(255,max(0,int(x[b,t,c])*int(gain[c])+int(bias[c]))) for t in range(x.shape[1])]
            low=min(a);out[b,c]=sum((v-low)**2 for v in a)
    elif n=='centered_gram_v2':
        T=x.shape[1]
        for b,c,d in np.ndindex(shape):
            sc=sum(int(x[b,t,c]) for t in range(T));sd=sum(int(x[b,t,d]) for t in range(T))
            out[b,c,d]=sum((T*int(x[b,t,c])-sc)*(T*int(x[b,t,d])-sd) for t in range(T))
    elif n=='weighted_distances_v2':
        y,weight=values[1:]
        for b,i,j in np.ndindex(shape):
            out[b,i,j]=sum(int(weight[d])*(int(x[b,i,d])-int(y[b,j,d]))**2 for d in range(x.shape[2]))
    elif n=='convolution_v2':
        f=values[1]
        for b,i,j,k in np.ndindex(shape):
            out[b,i,j,k]=sum(int(x[b,i+u,j+v,c])*int(f[k,u,v,c]) for u,v,c in np.ndindex(f.shape[1:]))
    elif n=='block_covariance_v2':
        for b,i,j,c,d in np.ndindex(shape):
            a=[int(x[b,2*i+u,2*j+v,c]) for u,v in np.ndindex((2,2))]
            z=[int(x[b,2*i+u,2*j+v,d]) for u,v in np.ndindex((2,2))]
            out[b,i,j,c,d]=sum((4*v-sum(a))*(4*w-sum(z)) for v,w in zip(a,z))
    else:
        mask,mix=values[1:];out=x.copy();B,H,W,C=x.shape
        for _ in range(3):
            old=out.copy()
            for b in range(B):
                for i in range(1,H-1):
                    for j in range(1,W-1):
                        if mask[i,j]:
                            local=[(4*int(old[b,i,j,c])+int(old[b,i-1,j,c])+int(old[b,i+1,j,c])+int(old[b,i,j-1,c])+int(old[b,i,j+1,c]))//8 for c in range(C)]
                            for d in range(C):out[b,i,j,d]=min(255,max(0,sum(v*int(mix[c,d]) for c,v in enumerate(local))))
    return out
