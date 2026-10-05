"""NumPy equivalence and wrong-answer rejection for the Astra audit repairs."""
import numpy as np
import pytest

from studies.user_study.adversarial.check import instrument_module
from studies.user_study.validate import python_function
from zinnia import ZKCircuit


def verify(body, shape=(3, 4), backend='mock'):
    source = f'def kernel(x: Integer[Array, "{" ".join(map(str, shape))}"]):\n{body}'
    values = [np.arange(np.prod(shape)).reshape(shape) - 5,
              np.zeros(shape, dtype=np.int64),
              np.random.default_rng(81425).integers(-8, 12, size=shape)]
    fn = python_function(source)
    expected = [np.asarray(fn(x.copy())) for x in values]
    ZKCircuit.from_source('kernel', source).compile()
    circuit = ZKCircuit.from_source('kernel', instrument_module(source, expected[0].shape))
    circuit.compile()
    for x, answer in zip(values, expected):
        proof = circuit.prove(x.copy(), answer, backend=backend)
        assert proof.proof_bytes_hex == 'mock_satisfied' if backend == 'mock' else circuit.verify(proof)
    wrong = expected[-1].copy()
    wrong.flat[0] += 1
    assert circuit.prove(values[-1].copy(), wrong, backend='mock').proof_bytes_hex != 'mock_satisfied'


@pytest.mark.parametrize('body', [
    '    return np.array([np.sum(x, where=x>0, initial=17)])\n',
    '    return np.sum(x, axis=1, where=x>0, initial=17, keepdims=True)\n',
    '    return np.prod(x, axis=1, where=x>0, initial=3)\n',
    '    return np.min(x, axis=1, where=x>0, initial=30)\n',
    '    return np.max(x, axis=1, where=x<0, initial=-30)\n',
    '    return np.minimum.reduce(x, 1, None, None, True, 30)\n',
    '    return np.sum(x, axis=1, dtype=float).astype(int) + 2\n',
    '    return x.astype("int64") * 2 + 3\n',
    '    return (x + 0).astype(float).astype(int) * 2\n',
    '''    out = np.full((4,3), -13, dtype=int)
    np.add(x, 7, out=out.T, where=x>0)
    return out
''',
    '''    out = np.zeros((4,3), dtype=int)
    np.sum(x, axis=0, out=out[:,1], initial=11)
    return out
''',
    '''    y = x.copy()
    for i in range(3):
        if x[i,0] < 0:
            continue
        y[i] = x[i] + 9
    return y
''',
    '''    y = x.copy()
    for i in range(3):
        if x[i,0] > 0:
            break
        np.add(y[i], 9, out=y[i])
    return y
''',
    '''    y = np.zeros_like(x)
    for i in range(3):
        indices = np.nonzero(x[i] > 0)[0]
        for j in indices:
            y[i,j] += 3
    return y
''',
    '''    return np.array([sum((r for r in x), np.zeros(4,dtype=int))])
''',
    '''    return np.array([row * 2 for i,row in enumerate(x)])
''',
    '''    return np.array([sum(column) for column in zip(*x)])
''',
    '''    rows = x.tolist()
    for i in range(3):
        for j in range(4):
            rows[i][j] += i + j
    return np.array(rows)
''',
    '''    a,b = np.split(x,2,axis=1)
    a[0,0] = 21
    return a + b + x[:, :2]
''',
    '    return (np.abs(x) // 2) + np.square(x)\n',
    '    return np.median(np.stack([x, np.zeros_like(x), np.full_like(x,5)]), axis=0).astype(int)\n',
])
def test_repaired_array_semantics(body):
    verify(body)


def test_module_instrumentation_preserves_helper_returns():
    source = '''import numpy as np
OFFSET = 3
def helper(row):
    return row * 2 + OFFSET
def kernel(x: Integer[Array, "2 3"]):
    return np.array([helper(row) for row in x])
'''
    values = np.arange(6).reshape(2,3)
    expected = python_function(source)(values)
    checked = ZKCircuit.from_source('kernel', instrument_module(source, expected.shape))
    checked.compile()
    assert checked.prove(values,expected,backend='mock').proof_bytes_hex == 'mock_satisfied'
    expected[0,0] += 1
    assert checked.prove(values,expected,backend='mock').proof_bytes_hex != 'mock_satisfied'


@pytest.mark.parametrize('backend', ['mock', 'halo2', 'halo2-base'])
@pytest.mark.parametrize('expression', [
    '(x.astype(float) // 2).astype(int)',
    '((x.astype(float) + 0.5) // -2).astype(int)',
    'np.floor_divide(x.astype(float), 2).astype(int)',
    'np.floor(x.astype(float) / 2).astype(int)',
    '(2 * ((x.astype(float) + 0.5) % -2)).astype(int)',
])
def test_float_floor_and_remainder(expression, backend):
    verify(f'    return {expression}\n', shape=(2,3), backend=backend)


@pytest.mark.parametrize('backend', ['mock', 'halo2', 'halo2-base'])
def test_smoothing_with_numpy_default_float_buffer(backend):
    verify('''    current = np.abs(x)
    for r in range(3):
        next_grid = np.zeros(current.shape)
        next_grid[:] = current
        for i in range(1, current.shape[0]-1):
            for j in range(1, current.shape[1]-1):
                next_grid[i,j] = (4*current[i,j]+current[i-1,j]+current[i+1,j]+current[i,j-1]+current[i,j+1]) // 8
        current = next_grid
    return current.astype(int)
''', backend=backend)
