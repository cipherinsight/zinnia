"""Regression probes for compiler-owned bounded dot/compress/take lowering.

Run after the coordinating agent registers and rebuilds the Rust module.
Positive oracles use ordinary NumPy. Every valid circuit also rejects a wrong
expected output; separate invalid-index and mismatched-length witnesses fail.
"""
import ast
import copy
import json
from pathlib import Path
import numpy as np
from zinnia import ZKCircuit
from studies.user_study.pilots.validate import instrument
from studies.user_study.validate import python_function

BODIES = {
    'dot_filtered': '''out = []
for row in x:
    active = row[row > floor]
    out.append(np.dot(active, active))
return np.array(out)''',
    'inner_filtered': '''out = []
for row in x:
    active = row[row > floor]
    out.append(np.inner(active, active))
return np.array(out)''',
    'compress_static_mask': 'return np.array([np.sum(np.compress(row > floor, row) ** 2) for row in x])',
    'compress_short_dynamic_mask': '''out = []
for row in x:
    selected = row[row > floor]
    out.append(np.sum(np.compress(selected != 0, row) ** 2))
return np.array(out)''',
    'take_static_source': '''out = []
for row in x:
    indices = np.flatnonzero(row > floor)
    active = np.take(row, indices)
    out.append(np.sum(active * active))
return np.array(out)''',
    'take_negative_indices': '''out = []
for row in x:
    indices = np.flatnonzero(row > floor) - row.shape[0]
    active = np.take(row, indices, axis=-1, mode='raise')
    out.append(np.sum(active * active))
return np.array(out)''',
    'take_dynamic_source': '''out = []
for row in x:
    active = row[row > floor]
    indices = np.flatnonzero(row > floor) * 0
    values = np.take(active, indices)
    out.append(np.sum(values * values))
return np.array(out)''',
    'take_dynamic_source_negative': '''out = []
for row in x:
    active = row[row > floor]
    indices = np.flatnonzero(row > floor) * 0 - 1
    values = np.take(active, indices)
    out.append(np.sum(values * values))
return np.array(out)''',
    'dot_different_capacity': '''out = []
for row in x:
    left = row[row > floor]
    first = row[:1]
    right = first[first > floor[:1]]
    out.append(np.dot(left, right))
return np.array(out)''',
    'compress_short_static': 'return np.array([np.sum(np.compress(row[:1] > floor[:1], row) ** 2) for row in x])',
}

def typed(body, shape):
    n, c = shape
    return f'def kernel(x: Integer[Array, "{n} {c}"], floor: Integer[Array, "{c}"]):\n' + ''.join('    '+line+'\n' for line in body.splitlines())

def positive(name, body, values):
    shape=values[0][0].shape
    source=typed(body,shape)
    reference=python_function(source)
    expected=[np.asarray(reference(*copy.deepcopy(case))) for case in values]
    original=ZKCircuit.from_source('kernel',source)
    original.compile()
    checked=ZKCircuit.from_source('kernel',instrument(source,(shape[0],)))
    checked.compile()
    for case, answer in zip(values,expected):
        proof=checked.prove(*copy.deepcopy(case),answer,backend='mock')
        assert proof.proof_bytes_hex=='mock_satisfied', name
    wrong=expected[-1].copy();wrong[0]+=1
    assert checked.prove(*copy.deepcopy(values[-1]),wrong,backend='mock').proof_bytes_hex!='mock_satisfied'
    return {'name':name,'status':'pass','positive_cases':len(values),'negative_control':True}

def main():
    floor=np.array([0,4,-3,7],dtype=np.int64)
    values=[]
    for row in (floor, floor+1, floor-1, np.array([5,4,-2,9]),np.array([-512,4096,-2,7])):
        values.append([np.stack([row,row[::-1]]),floor.copy()])
    rng=np.random.default_rng(717)
    for _ in range(12):
        values.append([rng.integers(-512,4097,size=(2,4),dtype=np.int64),rng.integers(-512,4097,size=4,dtype=np.int64)])
    rows=[]
    for name,body in BODIES.items():
        try:
            cases=values
            if name=='dot_different_capacity':
                cases=[[np.array([[2,-3,-4,-5],[0,-1,-2,-3]]),np.zeros(4,dtype=np.int64)], [np.full((2,4),-1,dtype=np.int64),np.zeros(4,dtype=np.int64)]]
            row=positive(name,body,cases)
        except BaseException as exc:
            if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
            row={'name':name,'status':'fail','error':f'{type(exc).__name__}: {exc}'}
        rows.append(row)
        print(json.dumps(row),flush=True)
    invalids={
        'take_out_of_bounds': 'return np.array([np.sum(np.take(row, np.flatnonzero(row > floor) + row.shape[0])) for row in x])',
        'dot_mismatched_length': 'return np.array([np.dot(row[row > floor], row[:1]) for row in x])',
    }
    for name,body in invalids.items():
        try:
            source=typed(body,(2,4))
            case=[np.full((2,4),4,dtype=np.int64),np.zeros(4,dtype=np.int64)]
            circuit=ZKCircuit.from_source('kernel',instrument(source,(2,)))
            circuit.compile()
            proof=circuit.prove(*case,np.zeros(2,dtype=np.int64),backend='mock')
            assert proof.proof_bytes_hex!='mock_satisfied'
            row={'name':name,'status':'pass','invalid_witness_rejected':True}
        except BaseException as exc:
            if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
            row={'name':name,'status':'fail','error':f'{type(exc).__name__}: {exc}'}
        rows.append(row)
        print(json.dumps(row),flush=True)
    Path(__file__).with_name('dynamic_api_probes.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert all(r['status']=='pass' for r in rows)

if __name__=='__main__':main()
