"""Bounded reduction shape/value checks before replacing rejection tests."""
from pathlib import Path
import json
import numpy as np
from zinnia import ZKCircuit


def main():
    rows=[]
    for method in ('sum','prod','min','max','any','all'):
        for axis in (None,0,1,-1,(0,1),()):
            expected_type = '1 1' if axis in (None,(0,1)) else '1 N<=3' if axis==0 else '2 N<=3' if axis==() else '2 1'
            for form in ('method','function'):
                expr=f'x.{method}(axis={axis!r},keepdims=True)' if form=='method' else f'np.{method}(x,axis={axis!r},keepdims=True)'
                source=f'''def foo(x: Int[Array, "2 N<=3"], expected: Int[Array, "{expected_type}"]):
    s = {expr}
    assert s.ndim == 2
    assert s.shape[0] == expected.shape[0]
    assert s.shape[1] == expected.shape[1]
    assert np.all(s == expected)
'''
                name=f'{method}/{axis!r}/{form}'
                try:
                    circuit=ZKCircuit.from_source('foo',source)
                    circuit.compile()
                    for values in (np.array([[-4],[-2]]),np.array([[0,3],[-1,0]]),np.array([[-4,0,7],[2,-3,4]])):
                        expected=getattr(np,method)(values,axis=axis,keepdims=True).astype(np.int64)
                        assert circuit.prove(values,expected,backend='mock').proof_bytes_hex=='mock_satisfied'
                    wrong=expected.copy();wrong.flat[0]+=1
                    assert circuit.prove(values,wrong,backend='mock').proof_bytes_hex!='mock_satisfied'
                    row={'name':name,'status':'pass','positive_cases':3,'negative_control':True}
                except BaseException as exc:
                    if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
                    row={'name':name,'status':'fail','error':f'{type(exc).__name__}: {exc}'}
                rows.append(row)
                if row['status']!='pass':print(json.dumps(row),flush=True)
    Path(__file__).with_name('keepdims_probes.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f'{sum(row["status"]=="pass" for row in rows)}/{len(rows)} passed')
    assert all(row['status']=='pass' for row in rows)

if __name__=='__main__':main()
