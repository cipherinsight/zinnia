"""Regressions discovered by the composed array study inventory."""
import numpy as np
import pytest
from zinnia import ZKCircuit
from studies.user_study.validate import checked_source

@pytest.mark.parametrize('left,right',[
 ((2,3,4),(2,4,2)), ((2,1,3,4),(1,5,4,2)),
 ((2,3,4),(4,2)), ((3,4),(2,4,2)),
 ((4,),(2,4,3)), ((2,3,4),(4,)),

])
@pytest.mark.parametrize('expression',['x @ y','np.matmul(x,y)'])
def test_batched_matmul(left,right,expression):
    x=np.arange(np.prod(left),dtype=np.int64).reshape(left)%11-5
    y=np.arange(np.prod(right),dtype=np.int64).reshape(right)%7-3
    expected=x@y
    ann=lambda shape: 'Integer[Array, "'+' '.join(map(str,shape))+'"]'
    source=f'def kernel(x: {ann(left)}, y: {ann(right)}):\n    return {expression}\n'
    program=ZKCircuit.from_source('kernel',checked_source(source,expected.shape)).compile()
    for sign in (1,-1):
        proof=program.prove(x*sign,y,expected*sign,backend='mock')
        assert proof.proof_bytes_hex=='mock_satisfied'

@pytest.mark.parametrize('expression',[
 'np.clip(x[:,1:-1,1:-1,:],0,255)',
 'np.where(x[:,1:-1,1:-1,:]>0, x[:,1:-1,1:-1,:], 0)',
 'np.minimum(x[:,1:-1,1:-1,:], 255)',
])
def test_empty_interior_shape(expression):
    source=f'''def kernel(x: Integer[Array, "2 1 2 3"]):
    y = {expression}
    assert y.shape == (2,0,0,3)
    return x
'''
    p=ZKCircuit.from_source('kernel',source).compile()
    assert p.prove(np.ones((2,1,2,3),dtype=np.int64),backend='mock').proof_bytes_hex=='mock_satisfied'

def test_invalid_batch_dimensions_rejected():
    with pytest.raises(Exception):
        ZKCircuit.from_source('kernel','''def kernel(x: Integer[Array, "2 3 4"], y: Integer[Array, "3 4 2"]):
    return x @ y
''').compile()

@pytest.mark.parametrize('expression,shape',[
 ('x[:, :0, :] @ y', (2,0,2)),
 ('x[:, :, :0] @ y[:0, :]', (2,3,2)),
])
def test_empty_matmul_intermediates(expression,shape):
    source=f"""def kernel(x: Integer[Array, "2 3 4"], y: Integer[Array, "4 2"]):
    a = {expression}
    assert a.shape == {shape}
    assert np.all(a == 0)
    return x
"""
    p=ZKCircuit.from_source('kernel',source).compile()
    assert p.prove(np.ones((2,3,4),dtype=np.int64),np.ones((4,2),dtype=np.int64),backend='mock').proof_bytes_hex=='mock_satisfied'

@pytest.mark.parametrize('count',[0,1,3])
def test_counted_loop_inside_private_branch_preserves_counter(count):
    source=f'''def kernel(x: Integer[Array, "2"]):
    counter = 9
    total = 0
    if x[0] > 0:
        counter = 0
        while counter < {count}:
            total += x[1]
            counter += 1
    return np.array([counter,total])
'''
    p=ZKCircuit.from_source('kernel',checked_source(source,(2,))).compile()
    for enabled in (-1,1):
        expected=np.array([count,count*7] if enabled>0 else [9,0])
        assert p.prove(np.array([enabled,7]),expected,backend='mock').proof_bytes_hex=='mock_satisfied'

@pytest.mark.parametrize('body',[
 'i = 0\nwhile i < 3:\n    i += 1\n    if i == 2: break',
 'i = 0\nwhile i < 3:\n    i += 1\n    continue',
 'n = 3\ni = 0\nwhile i < n:\n    n -= 1\n    i += 1',
 'i = 0\nwhile i < 2.5:\n    i += 1',
])
def test_noncanonical_loops_keep_python_semantics(body):
    # Counted `while` loops are unrolled by the IR generator itself; the
    # compiled counter must match CPython, and a wrong counter is rejected.
    source='def kernel(x: Integer[Array, "1"]):\n'+'\n'.join('    '+line for line in body.splitlines())+'\n    return np.array([i])\n'
    env={'np':np};exec(source.replace(': Integer[Array, "1"]',''),env);expected=int(env['kernel'](None)[0])
    p=ZKCircuit.from_source('kernel',checked_source(source,(1,))).compile()
    assert p.prove(np.array([0]),np.array([expected]),backend='mock').proof_bytes_hex=='mock_satisfied'
    assert p.prove(np.array([0]),np.array([expected+1]),backend='mock').proof_bytes_hex!='mock_satisfied'

@pytest.mark.parametrize('binding',['n = x.shape[0]', 'n = len(x)', 'n = 2'])
def test_masked_while_with_individual_dimension_binding(binding):
    source=f"""def kernel(x: Integer[Array, "2"]):
    {binding}
    total=0
    for outer in range(3):
        if x[0]>0:
            i=0
            while i<n:
                total+=x[i]
                i+=1
    return np.array([total])
"""
    p=ZKCircuit.from_source('kernel',checked_source(source,(1,))).compile()
    for sign in (-1,1):
        x=np.array([sign,5])
        expected=np.array([3*int(x.sum()) if sign>0 else 0])
        assert p.prove(x,expected,backend='mock').proof_bytes_hex=='mock_satisfied'
