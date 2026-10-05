"""Independent semantic probes for array_iteration, after root registers/builds it."""
import copy
import json
from pathlib import Path
import numpy as np
from zinnia import ZKCircuit
from studies.user_study.validate import python_function
from studies.user_study.adversarial.check import instrument_module as instrument

PROBES = {
    'split_column_alias': """
    y = x.copy()
    a,b = np.split(y,2,axis=1)
    a[0,0,0] = 21
    return a+b+y[:,:2,:]
""",
    'split_negative_boundary': """
    a,b = np.split(x,[-1],axis=1)
    return np.concatenate([b,a],axis=1)
""",
    'split_descending_boundary_alias': """
    y = x.copy()
    a,empty,c = np.split(y,[3,1],axis=1)
    a[0,1,0] = 51
    return np.concatenate([a,c],axis=1)
""",
    'split_array_boundaries': """
    a,b,c = np.split(x,np.array([1,3]),axis=1)
    return np.concatenate([c,b,a],axis=1)
""",
    'hsplit_alias': """
    y = x.copy()
    a,b = np.hsplit(y,2)
    a[1,1,0] = 28
    return y
""",
    'dsplit_alias': """
    y = x.copy()
    a,b = np.dsplit(y,2)
    a[1,1,0] = 28
    return y
""",
    'subsample_write_alias': """
    y = x.copy()
    view = y[:,::2,:]
    view[0,1,0] = 44
    return y
""",
    'subsample_source_alias': """
    y = x.copy()
    view = y[:,::2,1]
    y[1,2,1] = 37
    return view
""",
    'reduction_into_column_view': """
    out = np.zeros((4,2,3),dtype=int)
    np.sum(x,axis=0,out=out[:,:,1],initial=11)
    return out
""",
    'ufunc_into_strided_view': """
    out = np.zeros_like(x)
    np.add(x[:,::2,:],7,out=out[:,::2,:])
    return out
""",
    'static_ternary_dead_mutation': """
    y = x.copy()
    def change():
        y[0,0,0] = 99
        return y
    z = y if True else change()
    return z
""",
    'static_ternary_dead_mutation_false': """
    y = x.copy()
    def change():
        y[0,0,0] = 99
        return y
    z = change() if 0 else y
    return z
""",
    'static_ternary_dead_oob': '    return np.array([[x[i,j,0] if i == 2 else x[i+1,j,0] for j in range(4)] for i in range(3)])\n',
    'static_ternary_false_arm': '    return x[30,:,:] if 0 else x[:,:,0]\n',
    'ndindex_tuple': '''
    out = np.zeros_like(x)
    for i,j,k in np.ndindex(x.shape):
        out[i,j,k] = x[i,j,k] + i - j + k
    return out
''',
    'ndindex_args': '''
    out = np.zeros_like(x)
    for i,j,k in np.ndindex(3,4,2):
        out[i,j,k] = x[i,j,k]
    return out
''',
    'minimum_dtype_int': '    return np.minimum.accumulate(x,axis=0,dtype=int)\n',
    'maximum_dtype_numpy': '    return np.maximum.accumulate(x,axis=1,dtype=np.int64)\n',
    'minimum_dtype_attribute': '    return np.minimum.accumulate(x,axis=-1,dtype=x.dtype)\n',
    'maximum_dtype_string': '    return np.maximum.accumulate(x,axis=0,dtype="int64")\n',
    'reduceat_dtype_int': '    return np.add.reduceat(x,[0,2],axis=1,dtype=int)\n',
    'reduceat_dtype_numpy': '    return np.add.reduceat(x,[0,2],axis=1,dtype=np.int64)\n',
    'minimum_axis0': '    return np.minimum.accumulate(x, axis=0)\n',
    'maximum_negative_axis': '    return np.maximum.accumulate(x, axis=-1)\n',
    'minimum_out_alias': '''
    out = x.copy()
    np.minimum.accumulate(out, axis=1, out=out)
    return out
''',
    'reduceat_descending': '    return np.add.reduceat(x, np.array([2,0,1]), axis=0)\n',
    'reduceat_repeated_negative_axis': '    return np.add.reduceat(x, [1,1,0], axis=-1)\n',
    'reduceat_noncontiguous': '    return np.add.reduceat(x[:,::2,:], [0,1], axis=1)\n',
    'reduceat_out': '''
    out = np.zeros((3,2,2),dtype=int)
    np.add.reduceat(x,[0,2],axis=1,out=out)
    return out
''',
    'window_axes': '    return np.lib.stride_tricks.sliding_window_view(x, (2,2),axis=(0,1)).sum(axis=(-1,-2))\n',
    'window_repeated_axes': '    return np.lib.stride_tricks.sliding_window_view(x,(2,2),axis=(1,1)).sum(axis=(-1,-2))\n',
    'window_noncontiguous': '    return np.lib.stride_tricks.sliding_window_view(x[:,::2,:],2,axis=0).sum(axis=-1)\n',
    'window_alias': '''
    y = x.copy()
    w = np.lib.stride_tricks.sliding_window_view(y,(2,2),axis=(0,1))
    y[0,0,0] += 31
    return w.sum(axis=(-1,-2))
''',
    'roll_axis_none': '    return np.roll(x,-37)\n',
    'roll_repeated_axes': '    return np.roll(x,(9,-7,2),axis=(0,0,-1))\n',
    'roll_scalar_broadcast': '    return np.roll(x,2,axis=(0,1,-1))\n',
    'roll_shift_broadcast': '    return np.roll(x,(2,-3),axis=1)\n',
    'roll_noncontiguous': '    return np.roll(x[:,::2,:],1,axis=1)\n',
    'pad_scalar': '    return np.pad(x,1,mode="edge")\n',
    'pad_asymmetric_axes': '    return np.pad(x,((0,1),(2,0),(1,2)),mode="edge")\n',
    'pad_broadcast_pair': '    return np.pad(x,(1,2),mode="edge")\n',
    'pad_noncontiguous': '    return np.pad(x[:,::2,:],((1,0),),mode="edge")\n',
}

INVALID_PROBES = {
    'scan_converting_dtype': '    return np.minimum.accumulate(x,axis=0,dtype=float)\n',
    'reduceat_narrow_string_dtype': '    return np.add.reduceat(x,[0,2],axis=1,dtype="int8")\n',
    'split_negative_sections': '    return np.split(x,-2,axis=1)[0]\n',
    'static_ternary_live_oob': '    return x[30,:,:] if 1 else x[:,:,0]\n',
    'negative_ndindex': '    return np.array(list(np.ndindex(-1,2)))\n',
    'scan_axis_oob': '    return np.minimum.accumulate(x,axis=3)\n',
    'scan_tuple_axis': '    return np.maximum.accumulate(x,axis=(0,1))\n',
    'reduceat_negative_index': '    return np.add.reduceat(x,[-1],axis=0)\n',
    'reduceat_index_oob': '    return np.add.reduceat(x,[3],axis=0)\n',
    'window_too_large': '    return np.lib.stride_tricks.sliding_window_view(x,5,axis=1)\n',
    'window_axis_mismatch': '    return np.lib.stride_tricks.sliding_window_view(x,(2,2),axis=0)\n',
    'window_writeable': '    return np.lib.stride_tricks.sliding_window_view(x,2,axis=0,writeable=True)\n',
    'roll_axis_oob': '    return np.roll(x,1,axis=3)\n',
    'pad_negative': '    return np.pad(x,-1,mode="edge")\n',
}

def main():
    rng = np.random.default_rng(84217)
    fixtures = [np.arange(24,dtype=np.int64).reshape(3,4,2)-12,
                rng.integers(-512,4097,size=(3,4,2),dtype=np.int64),
                np.where(np.arange(24).reshape(3,4,2)%2,4096,-512)]
    results=[]
    for name,body in PROBES.items():
        source='def kernel(x: Integer[Array, "3 4 2"]):\n'+body
        row={'name':name,'status':'fail'}
        try:
            ref=python_function(source)
            expected=[np.asarray(ref(copy.deepcopy(x))) for x in fixtures]
            ZKCircuit.from_source('kernel',source).compile()
            checked=ZKCircuit.from_source('kernel',instrument(source,expected[0].shape))
            checked.compile()
            for x,answer in zip(fixtures,expected):
                p=checked.prove(copy.deepcopy(x),answer,backend='mock')
                assert p.proof_bytes_hex=='mock_satisfied','oracle rejected'
            wrong=expected[-1].copy(); wrong.flat[0]+=1
            p=checked.prove(copy.deepcopy(fixtures[-1]),wrong,backend='mock')
            assert p.proof_bytes_hex!='mock_satisfied','wrong answer accepted'
            row.update(status='pass',fixtures=len(fixtures),negative_control=True)
        except Exception as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
        print(name,row,flush=True)
        results.append(row)
    for name,body in INVALID_PROBES.items():
        source='def kernel(x: Integer[Array, "3 4 2"]):\n'+body
        row={'name':name,'status':'fail','kind':'argument_rejection'}
        try:
            ZKCircuit.from_source('kernel',source).compile()
            row['error']='invalid or explicitly unsupported argument accepted'
        except Exception as exc:
            row.update(status='pass',error=f'{type(exc).__name__}: {exc}')
        print(name,row,flush=True)
        results.append(row)
    Path(__file__).with_name('api-probes.json').write_text(json.dumps(results,indent=2)+'\n')
    raise SystemExit(not all(r['status']=='pass' for r in results))

if __name__=='__main__':main()
