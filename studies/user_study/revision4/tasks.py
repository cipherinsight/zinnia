"""Current six-task inventory; only the changed contract receives a new ID."""
import ast
import copy
from dataclasses import replace
import numpy as np
from ..revision3 import tasks as previous
from ..revision2.tasks import body, WhileLoops
from ..tasks import chained_indices

FILTER='strided_filters_v4'
old=next(t for t in previous.TASKS if t.name=='strided_filters_v3')
# Change dilation, not output stride. Regenerate syntax-derived variants from
# the revised loop body so a chained/while spelling cannot retain old offsets.
variants={}
for name,code in old.variants.items():
    if name in ('chained_indexing','default_zeros','while_loops'):continue
    code=code.replace('H-2*(U-1)-1','H-3*(U-1)-1').replace('W-2*(V-1)-1','W-3*(V-1)-1')
    code=code.replace('2*i:2*i+2*U:2','2*i:2*i+3*U:3').replace('2*j:2*j+2*V:2','2*j:2*j+3*V:3')
    code=code.replace('2*u:2*u+2*OH:2','3*u:3*u+2*OH:2').replace('2*v:2*v+2*OW:2','3*v:3*v+2*OW:2')
    code=code.replace('2*i+2*u','2*i+3*u').replace('2*j+2*v','2*j+3*v')
    variants[name]=code
variants['chained_indexing']=chained_indices(variants['nested_loops'])
variants['default_zeros']=variants['nested_loops'].replace(',dtype=int','')
variants['while_loops']=ast.unparse(ast.fix_missing_locations(WhileLoops().visit(ast.parse(variants['nested_loops']))))
WINDOWS=body('''
 B,H,W,C=x.shape
 K,U,V,C=filters.shape
 windows=np.lib.stride_tricks.sliding_window_view(x,(3*(U-1)+1,3*(V-1)+1),axis=(1,2))
 patches=windows[:,::2,::2,:,::3,::3]
''')+'\n'
variants['window_einsum']=WINDOWS+"return np.einsum('bijcuv,kuvc->bijk',patches,filters)"
variants['window_import']="from numpy.lib.stride_tricks import sliding_window_view as windows_view\n"+variants['window_einsum'].replace('np.lib.stride_tricks.sliding_window_view','windows_view')
variants['window_module_alias']="import numpy.lib.stride_tricks as strides\n"+variants['window_einsum'].replace('np.lib.stride_tricks.sliding_window_view','strides.sliding_window_view')
variants['window_tensordot']=WINDOWS+"return np.tensordot(patches,filters,axes=([3,4,5],[3,1,2]))"
variants['window_matmul']=WINDOWS+body('''
 OH=patches.shape[1]
 OW=patches.shape[2]
 rows=patches.transpose(0,1,2,4,5,3).reshape(B*OH*OW,U*V*C)
 return (rows @ filters.reshape(K,U*V*C).T).reshape(B,OH,OW,K)
''')
variants['patch_helper']=body('''
 def apply_patch(patch,coefficients):
     return np.einsum('buvc,kuvc->bk',patch,coefficients)
''')+'\n'+variants['patch_einsum'].replace("np.einsum('buvc,kuvc->bk',patch,filters)",'apply_patch(patch,filters)')
variants['empty_output_buffer']=variants['patch_sum'].replace('np.zeros','np.empty')
variants['comprehension']=body('''
 B,H,W,C=x.shape
 K,U,V,C=filters.shape
 OH=(H-3*(U-1)-1)//2+1
 OW=(W-3*(V-1)-1)//2+1
 return np.array([[[[sum(x[b,2*i+3*u,2*j+3*v,c]*filters[k,u,v,c] for u in range(U) for v in range(V) for c in range(C)) for k in range(K)] for j in range(OW)] for i in range(OH)] for b in range(B)])
''')
FILTER_TASK=replace(old,name=FILTER,profiles=(
 dict(B=1,H=8,W=10,C=2,K=3,U=2,V=3),
 dict(B=1,H=1,W=1,C=1,K=1,U=1,V=1),
 dict(B=2,H=9,W=7,C=3,K=2,U=3,V=2),
 dict(B=2,H=4,W=7,C=2,K=1,U=2,V=3)),variants=variants)
TASKS=[FILTER_TASK if t.name==old.name else copy.deepcopy(t) for t in previous.TASKS]

def signatures(task,p):
    if task.name!=FILTER:return previous.signatures(task,p)
    return {'x':(p['B'],p['H'],p['W'],p['C']),'filters':(p['K'],p['U'],p['V'],p['C'])},(p['B'],(p['H']-3*(p['U']-1)-1)//2+1,(p['W']-3*(p['V']-1)-1)//2+1,p['K'])

def fixtures(task,p):
    # Input layouts and numeric domains are unchanged; only the output mapping
    # uses the new dilation. Original fixture generation remains versioned.
    return previous.fixtures(replace(task,name=old.name) if task.name==FILTER else task,p)

def oracle(task,p,values):
    if task.name!=FILTER:return previous.oracle(task,p,values)
    x,filters=values
    _,shape=signatures(task,p)
    out=np.zeros(shape,dtype=np.int64)
    for b,i,j,k in np.ndindex(shape):
        out[b,i,j,k]=sum(int(x[b,2*i+3*u,2*j+3*v,c])*int(filters[k,u,v,c]) for u,v,c in np.ndindex(filters.shape[1:]))
    return out

def source(task,variant,profile):
    shapes,_=signatures(task,profile)
    args=', '.join(f'{n}: Integer[Array, "{" ".join(map(str,s))}"]' for n,s in shapes.items())
    return 'def kernel('+args+'):\n'+'\n'.join('    '+s for s in task.variants[variant].splitlines())+'\n'
