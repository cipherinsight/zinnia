"""Version 3: explicit batch, projection, window, and block relationships."""
import ast
import numpy as np
from ..revision2.tasks import Task, body, WhileLoops
from ..tasks import chained_indices

TASKS=[]
def add(name, profiles, args, rank, variants):
    variants['chained_indexing']=chained_indices(variants['nested_loops'])
    variants['default_zeros']=variants['nested_loops'].replace(', dtype=int','').replace(',dtype=int','')
    variants['while_loops']=ast.unparse(ast.fix_missing_locations(WhileLoops().visit(ast.parse(variants['nested_loops']))))
    TASKS.append(Task(name,tuple(profiles),tuple(args),rank,variants))

add('channel_energy_v3',[dict(B=2,T=3,C=2),dict(B=1,T=1,C=1),dict(B=3,T=2,C=4)],['x','gain','bias'],2,{
'broadcast': 'a = np.clip(x*gain[:,None,:]+bias[:,None,:],0,255)\nz = a-a.min(axis=1,keepdims=True)\nreturn (z*z).sum(axis=1)',
'where': 'a = x*gain[:,None,:]+bias[:,None,:]\na = np.where(a<0,0,np.where(a>255,255,a))\nz = a-a.min(axis=1)[:,None,:]\nreturn np.einsum("btc,btc->bc",z,z)',
'negative_axis': 'a = (x*gain[:,None,:]+bias[:,None,:]).clip(0,255)\na -= a.min(axis=-2,keepdims=True)\nreturn (a*a).sum(axis=-2)',
'per_batch': body('''
 result=[]
 for b in range(x.shape[0]):
     a=np.clip(x[b]*gain[b]+bias[b],0,255)
     a -= a.min(axis=0)
     result.append((a*a).sum(axis=0))
 return np.array(result)
'''),
'column_slices': body('''
 B,T,C=x.shape
 out=np.zeros((B,C),dtype=int)
 for b in range(B):
     for c in range(C):
         a=np.clip(x[b,:,c]*gain[b,c]+bias[b,c],0,255)
         a=a-min(a)
         out[b,c]=sum(a*a)
 return out
'''),
'comprehension': body('''
 result=[]
 for b in range(x.shape[0]):
     row=[]
     for c in range(x.shape[2]):
         values=[min(255,max(0,v*gain[b,c]+bias[b,c])) for v in x[b,:,c]]
         low=min(values)
         row.append(sum((v-low)**2 for v in values))
     result.append(row)
 return np.array(result)
'''),
'nested_loops': body('''
 B,T,C=x.shape
 out=np.zeros((B,C),dtype=int)
 for b in range(B):
     for c in range(C):
         low=255
         for t in range(T):
             low=min(low,min(255,max(0,x[b,t,c]*gain[b,c]+bias[b,c])))
         for t in range(T):
             delta=min(255,max(0,x[b,t,c]*gain[b,c]+bias[b,c]))-low
             out[b,c]+=delta*delta
 return out
''')})
add('cross_products_v3',[dict(B=2,T=3,C=2,D=4),dict(B=1,T=1,C=1,D=1),dict(B=3,T=2,C=4,D=2)],['x','y'],3,{
'einsum': 'T=x.shape[1]\nz=T*x-x.sum(axis=1,keepdims=True)\nw=T*y-y.sum(axis=1,keepdims=True)\nreturn np.einsum("btc,btd->bcd",z,w)',
'batched_matmul': 'T=x.shape[1]\nz=T*x-x.sum(axis=1)[:,None,:]\nw=T*y-y.sum(axis=1)[:,None,:]\nreturn z.transpose(0,2,1) @ w',
'broadcast': 'T=x.shape[1]\nz=T*x-x.sum(axis=1,keepdims=True)\nw=T*y-y.sum(axis=1,keepdims=True)\nreturn (z[:,:,:,None]*w[:,:,None,:]).sum(axis=1)',
'algebraic_identity': 'T=x.shape[1]\ns=x.sum(axis=1)\nr=y.sum(axis=1)\nreturn T*T*np.einsum("btc,btd->bcd",x,y)-T*s[:,:,None]*r[:,None,:]',
'per_batch_dot': body('''
 result=[]
 T=x.shape[1]
 for b in range(x.shape[0]):
     z=T*x[b]-x[b].sum(axis=0)
     w=T*y[b]-y[b].sum(axis=0)
     result.append(np.dot(z.T,w))
 return np.array(result)
'''),
'per_batch_tensordot': body('''
 T=x.shape[1]
 z=T*x-x.sum(axis=1,keepdims=True)
 w=T*y-y.sum(axis=1,keepdims=True)
 return np.array([np.tensordot(z[b],w[b],axes=([0],[0])) for b in range(x.shape[0])])
'''),
'nested_loops': body('''
 B,T,C=x.shape
 D=y.shape[2]
 out=np.zeros((B,C,D),dtype=int)
 for b in range(B):
     for c in range(C):
         for d in range(D):
             sc=0
             sd=0
             for t in range(T):
                 sc+=x[b,t,c]
                 sd+=y[b,t,d]
             for t in range(T):
                 out[b,c,d]+=(T*x[b,t,c]-sc)*(T*y[b,t,d]-sd)
 return out
''')})
add('projected_distances_v3',[dict(B=2,N=3,M=2,D=2,F=3),dict(B=1,N=1,M=1,D=1,F=1),dict(B=1,N=2,M=4,D=3,F=2)],['x','y','transform','weight'],3,{
'broadcast': 'a=x @ transform\nb=y @ transform\ndelta=a[:,:,None,:]-b[:,None,:,:]\nreturn (delta*delta*weight[:,None,None,:]).sum(axis=-1)',
'einsum': 'a=np.einsum("bnd,bdf->bnf",x,transform)\nb=np.einsum("bmd,bdf->bmf",y,transform)\ndelta=a[:,:,None,:]-b[:,None,:,:]\nreturn np.einsum("bnmf,bf,bnmf->bnm",delta,weight,delta)',
'matmul_identity': 'a=np.matmul(x,transform)\nb=np.matmul(y,transform)\nreturn (a*a*weight[:,None,:]).sum(axis=2)[:,:,None]+(b*b*weight[:,None,:]).sum(axis=2)[:,None,:]-2*((a*weight[:,None,:]) @ b.transpose(0,2,1))',
'projection_broadcast': 'a=(x[:,:,:,None]*transform[:,None,:,:]).sum(axis=2)\nb=(y[:,:,:,None]*transform[:,None,:,:]).sum(axis=2)\nreturn ((a[:,:,None,:]-b[:,None,:,:])**2*weight[:,None,None,:]).sum(axis=3)',
'per_batch_dot': body('''
 result=[]
 for b in range(x.shape[0]):
     a=np.dot(x[b],transform[b])
     other=np.dot(y[b],transform[b])
     delta=a[:,None,:]-other[None,:,:]
     result.append((delta*delta*weight[b]).sum(axis=2))
 return np.array(result)
'''),
'comprehension': 'return np.array([[[sum(weight[b,f]*sum((x[b,i,d]-y[b,j,d])*transform[b,d,f] for d in range(x.shape[2]))**2 for f in range(weight.shape[1])) for j in range(y.shape[1])] for i in range(x.shape[1])] for b in range(x.shape[0])])',
'nested_loops': body('''
 B,N,D=x.shape
 M=y.shape[1]
 F=weight.shape[1]
 a=np.zeros((B,N,F),dtype=int)
 other=np.zeros((B,M,F),dtype=int)
 for b in range(B):
     for f in range(F):
         for d in range(D):
             for i in range(N):
                 a[b,i,f]+=x[b,i,d]*transform[b,d,f]
             for j in range(M):
                 other[b,j,f]+=y[b,j,d]*transform[b,d,f]
 out=np.zeros((B,N,M),dtype=int)
 for b in range(B):
     for i in range(N):
         for j in range(M):
             for f in range(F):
                 delta=a[b,i,f]-other[b,j,f]
                 out[b,i,j]+=weight[b,f]*delta*delta
 return out
'''),
'project_differences': body('''
 B,N,D=x.shape
 M=y.shape[1]
 F=weight.shape[1]
 out=np.zeros((B,N,M),dtype=int)
 for b in range(B):
     for i in range(N):
         for j in range(M):
             for f in range(F):
                 delta=0
                 for d in range(D):
                     delta+=(x[b,i,d]-y[b,j,d])*transform[b,d,f]
                 out[b,i,j]+=weight[b,f]*delta*delta
 return out
''')})
CONV='B,H,W,C=x.shape\nK,U,V,C=filters.shape\nOH=(H-2*(U-1)-1)//2+1\nOW=(W-2*(V-1)-1)//2+1\nout=np.zeros((B,OH,OW,K),dtype=int)\n'
add('strided_filters_v3',[dict(B=1,H=5,W=7,C=2,K=3,U=2,V=3),dict(B=1,H=1,W=1,C=1,K=1,U=1,V=1),dict(B=2,H=7,W=6,C=3,K=2,U=3,V=2)],['x','filters'],4,{
'patch_einsum': CONV+body('''
 for i in range(OH):
     for j in range(OW):
         patch=x[:,2*i:2*i+2*U:2,2*j:2*j+2*V:2,:]
         out[:,i,j,:]=np.einsum('buvc,kuvc->bk',patch,filters)
 return out
'''),
'offset_tensordot':CONV+body('''
 for u in range(U):
     for v in range(V):
         out+=np.tensordot(x[:,2*u:2*u+2*OH:2,2*v:2*v+2*OW:2,:],filters[:,u,v,:],axes=([-1],[-1]))
 return out
'''),
'offset_matmul':CONV+body('''
 for u in range(U):
     for v in range(V):
         out+=x[:,2*u:2*u+2*OH:2,2*v:2*v+2*OW:2,:] @ filters[:,u,v,:].T
 return out
'''),
'patch_broadcast':CONV+body('''
 for i in range(OH):
     for j in range(OW):
         patch=x[:,2*i:2*i+2*U:2,2*j:2*j+2*V:2,:]
         out[:,i,j,:]=(patch[:,None,:,:,:]*filters[None,:,:,:,:]).sum(axis=(2,3,4))
 return out
'''),
'flatten_dot':CONV+body('''
 f=filters.reshape(K,-1)
 for i in range(OH):
     for j in range(OW):
         patch=x[:,2*i:2*i+2*U:2,2*j:2*j+2*V:2,:].reshape(B,-1)
         out[:,i,j,:]=patch @ f.T
 return out
'''),
'patch_sum':CONV+body('''
 for b in range(B):
     for i in range(OH):
         for j in range(OW):
             for k in range(K):
                 out[b,i,j,k]=np.sum(x[b,2*i:2*i+2*U:2,2*j:2*j+2*V:2,:]*filters[k])
 return out
'''),
'nested_loops':CONV+body('''
 for b in range(B):
     for i in range(OH):
         for j in range(OW):
             for k in range(K):
                 for u in range(U):
                     for v in range(V):
                         for c in range(C):
                             out[b,i,j,k]+=x[b,2*i+2*u,2*j+2*v,c]*filters[k,u,v,c]
 return out
''')})
STACK='a=np.stack([x[:,0::2,0::3,:],x[:,0::2,1::3,:],x[:,0::2,2::3,:],x[:,1::2,0::3,:],x[:,1::2,1::3,:],x[:,1::2,2::3,:]],axis=3)\n'
add('rectangular_covariance_v3',[dict(B=1,H=4,W=9,C=2),dict(B=1,H=2,W=3,C=1),dict(B=2,H=2,W=6,C=3)],['x'],5,{
'reshape_einsum': 'B,H,W,C=x.shape\na=x.reshape(B,H//2,2,W//3,3,C).transpose(0,1,3,2,4,5).reshape(B,H//2,W//3,6,C)\nz=6*a-a.sum(axis=3,keepdims=True)\nreturn np.einsum("bhwpc,bhwpd->bhwcd",z,z)',
'stack_matmul':STACK+'z=6*a-a.sum(axis=3,keepdims=True)\nreturn z.transpose(0,1,2,4,3) @ z',
'broadcast':STACK+'z=6*a-a.sum(axis=3,keepdims=True)\nreturn (z[:,:,:,:,:,None]*z[:,:,:,:,None,:]).sum(axis=3)',
'algebraic_identity':STACK+'s=a.sum(axis=3)\nreturn 36*np.einsum("bhwpc,bhwpd->bhwcd",a,a)-6*s[:,:,:,:,None]*s[:,:,:,None,:]',
'block_dot':body('''
 B,H,W,C=x.shape
 out=np.zeros((B,H//2,W//3,C,C),dtype=int)
 for b in range(B):
     for i in range(H//2):
         for j in range(W//3):
             a=x[b,2*i:2*i+2,3*j:3*j+3,:].reshape(6,C)
             z=6*a-a.sum(axis=0)
             out[b,i,j,:,:]=np.dot(z.T,z)
 return out
'''),
'nested_loops':body('''
 B,H,W,C=x.shape
 out=np.zeros((B,H//2,W//3,C,C),dtype=int)
 for b in range(B):
     for i in range(H//2):
         for j in range(W//3):
             for c in range(C):
                 for d in range(C):
                     sc=0
                     sd=0
                     for u in range(2):
                         for v in range(3):
                             sc+=x[b,2*i+u,3*j+v,c]
                             sd+=x[b,2*i+u,3*j+v,d]
                     for u in range(2):
                         for v in range(3):
                             out[b,i,j,c,d]+=(6*x[b,2*i+u,3*j+v,c]-sc)*(6*x[b,2*i+u,3*j+v,d]-sd)
 return out
''')})
add('batch_diffusion_v3',[dict(B=2,H=4,W=5,C=2),dict(B=1,H=1,W=2,C=1),dict(B=2,H=3,W=4,C=3)],['x','mask','mix'],4,{
'slices_matmul':body('''
 a=x.copy()
 for step in range(3):
     out=a.copy()
     local=(4*a[:,1:-1,1:-1,:]+a[:,:-2,1:-1,:]+a[:,2:,1:-1,:]+a[:,1:-1,:-2,:]+a[:,1:-1,2:,:])//8
     value=np.clip(local @ mix[:,None,:,:],0,255)
     out[:,1:-1,1:-1,:]=np.where(mask[:,1:-1,1:-1,None]!=0,value,a[:,1:-1,1:-1,:])
     a=out
 return a
'''),
'slices_einsum':body('''
 a=np.copy(x)
 for step in range(3):
     out=np.copy(a)
     local=(4*a[:,1:-1,1:-1,:]+a[:,:-2,1:-1,:]+a[:,2:,1:-1,:]+a[:,1:-1,:-2,:]+a[:,1:-1,2:,:])//8
     value=np.clip(np.einsum('bhwc,bcd->bhwd',local,mix),0,255)
     out[:,1:-1,1:-1,:]=np.where(mask[:,1:-1,1:-1,None]!=0,value,a[:,1:-1,1:-1,:])
     a=out
 return a
'''),
'cell_dot':body('''
 a=x.copy()
 for step in range(3):
     out=a.copy()
     for b in range(a.shape[0]):
         for i in range(1,a.shape[1]-1):
             for j in range(1,a.shape[2]-1):
                 if mask[b,i,j]!=0:
                     local=(4*a[b,i,j]+a[b,i-1,j]+a[b,i+1,j]+a[b,i,j-1]+a[b,i,j+1])//8
                     out[b,i,j]=np.clip(np.dot(local,mix[b]),0,255)
     a=out
 return a
'''),
'nested_loops':body('''
 B,H,W,C=x.shape
 a=x.copy()
 for step in range(3):
     out=a.copy()
     for b in range(B):
         for i in range(1,H-1):
             for j in range(1,W-1):
                 if mask[b,i,j]!=0:
                     for d in range(C):
                         total=0
                         for c in range(C):
                             local=(4*a[b,i,j,c]+a[b,i-1,j,c]+a[b,i+1,j,c]+a[b,i,j-1,c]+a[b,i,j+1,c])//8
                             total+=local*mix[b,c,d]
                         out[b,i,j,d]=min(255,max(0,total))
     a=out
 return a
''')})
TASKS[-1].variants['buffer_assignment']=TASKS[-1].variants['slices_einsum'].replace('a=out','a[:,:,:,:]=out')
TASKS[-1].variants['read_copy_write_original']=TASKS[-1].variants['nested_loops'].replace('out=a.copy()','old=a.copy()\n    out=a').replace('4*a[','4*old[').replace('+a[','+old[')

def signatures(task,p):
 n=task.name
 if n=='channel_energy_v3':return {'x':(p['B'],p['T'],p['C']),'gain':(p['B'],p['C']),'bias':(p['B'],p['C'])},(p['B'],p['C'])
 if n=='cross_products_v3':return {'x':(p['B'],p['T'],p['C']),'y':(p['B'],p['T'],p['D'])},(p['B'],p['C'],p['D'])
 if n=='projected_distances_v3':return {'x':(p['B'],p['N'],p['D']),'y':(p['B'],p['M'],p['D']),'transform':(p['B'],p['D'],p['F']),'weight':(p['B'],p['F'])},(p['B'],p['N'],p['M'])
 if n=='strided_filters_v3':return {'x':(p['B'],p['H'],p['W'],p['C']),'filters':(p['K'],p['U'],p['V'],p['C'])},(p['B'],(p['H']-2*(p['U']-1)-1)//2+1,(p['W']-2*(p['V']-1)-1)//2+1,p['K'])
 if n=='rectangular_covariance_v3':return {'x':(p['B'],p['H'],p['W'],p['C'])},(p['B'],p['H']//2,p['W']//3,p['C'],p['C'])
 return {'x':(p['B'],p['H'],p['W'],p['C']),'mask':(p['B'],p['H'],p['W']),'mix':(p['B'],p['C'],p['C'])},(p['B'],p['H'],p['W'],p['C'])

def source(task,variant,profile):
 shapes,_=signatures(task,profile)
 args=', '.join(f'{n}: Integer[Array, "{" ".join(map(str,s))}"]' for n,s in shapes.items())
 return 'def kernel('+args+'):\n'+'\n'.join('    '+s for s in task.variants[variant].splitlines())+'\n'

def fixtures(task,profile):
 shapes,_=signatures(task,profile);rng=np.random.default_rng(20260923);result=[]
 for mode in range(8):
  values=[]
  for name,shape in shapes.items():
   if mode==0:a=np.zeros(shape,dtype=np.int64)
   elif mode==1:a=np.full(shape,3,dtype=np.int64)
   elif mode==2:a=np.arange(np.prod(shape),dtype=np.int64).reshape(shape)%17-8
   elif mode==3:a=np.full(shape,-3,dtype=np.int64)
   elif mode==4:a=np.resize(np.array([-16,16,0,1,-1],dtype=np.int64),shape)
   else:a=rng.integers(-16,17,size=shape,dtype=np.int64)
   if name in ('weight','gain'):a=np.abs(a)%5
   if name=='transform':a=a%9-4
   if task.name=='channel_energy_v3' and mode==4:
    if name=='x':a=np.resize(np.array([-64,64,0,63],dtype=np.int64),shape)
    elif name=='gain':a=np.full(shape,4,dtype=np.int64)
    elif name=='bias':a=np.full(shape,16,dtype=np.int64)
   if task.name=='batch_diffusion_v3':
    if name=='x':a=np.abs(a)*7
    elif name=='mask':a=np.abs(a)%2
    elif name=='mix':a=a%5-2
    if mode==1 and name=='mix':a=np.broadcast_to(np.eye(shape[-1],dtype=np.int64),shape).copy()
    if mode==3 and name=='mask':a=np.ones(shape,dtype=np.int64)
   values.append(a)
  result.append(values)
 return result

def oracle(task,p,values):
 """Independent scalar algorithms, with Python integer intermediates."""
 x=values[0];_,shape=signatures(task,p);out=np.zeros(shape,dtype=np.int64);n=task.name
 if n=='channel_energy_v3':
  gain,bias=values[1:]
  for b,c in np.ndindex(shape):
   a=[min(255,max(0,int(x[b,t,c])*int(gain[b,c])+int(bias[b,c]))) for t in range(x.shape[1])]
   low=min(a);out[b,c]=sum((v-low)**2 for v in a)
 elif n=='cross_products_v3':
  y=values[1];T=x.shape[1]
  for b,c,d in np.ndindex(shape):
   sc=sum(int(x[b,t,c]) for t in range(T));sd=sum(int(y[b,t,d]) for t in range(T))
   out[b,c,d]=sum((T*int(x[b,t,c])-sc)*(T*int(y[b,t,d])-sd) for t in range(T))
 elif n=='projected_distances_v3':
  y,transform,weight=values[1:]
  for b,i,j in np.ndindex(shape):
   for f in range(weight.shape[1]):
    left=sum(int(x[b,i,d])*int(transform[b,d,f]) for d in range(x.shape[2]))
    right=sum(int(y[b,j,d])*int(transform[b,d,f]) for d in range(y.shape[2]))
    out[b,i,j]+=int(weight[b,f])*(left-right)**2
 elif n=='strided_filters_v3':
  filters=values[1]
  for b,i,j,k in np.ndindex(shape):
   out[b,i,j,k]=sum(int(x[b,2*i+2*u,2*j+2*v,c])*int(filters[k,u,v,c]) for u,v,c in np.ndindex(filters.shape[1:]))
 elif n=='rectangular_covariance_v3':
  for b,i,j,c,d in np.ndindex(shape):
   a=[int(x[b,2*i+u,3*j+v,c]) for u,v in np.ndindex((2,3))]
   z=[int(x[b,2*i+u,3*j+v,d]) for u,v in np.ndindex((2,3))]
   out[b,i,j,c,d]=sum((6*v-sum(a))*(6*w-sum(z)) for v,w in zip(a,z))
 else:
  mask,mix=values[1:];out=x.copy();B,H,W,C=x.shape
  for _ in range(3):
   old=out.copy()
   for b in range(B):
    for i in range(1,H-1):
     for j in range(1,W-1):
      if mask[b,i,j]:
       local=[(4*int(old[b,i,j,c])+int(old[b,i-1,j,c])+int(old[b,i+1,j,c])+int(old[b,i,j-1,c])+int(old[b,i,j+1,c]))//8 for c in range(C)]
       for d in range(C):out[b,i,j,d]=min(255,max(0,sum(v*int(mix[b,c,d]) for c,v in enumerate(local))))
 return out
