"""Matched, untimed matrix practice for both crossover environments."""
import numpy as np

IDS = ('practice_zinnia_v2', 'practice_rust_v2')
TITLES = {task: ['Practice: sum above-baseline readings', '练习：汇总超出基准的读数'] for task in IDS}
RANKS = dict.fromkeys(IDS, [2, 1])
OUT_RANK = dict.fromkeys(IDS, 1)


def names(task):
    return ['x', 'baseline']


def starter(task, condition):
    if condition == 'zinnia':
        return 'def kernel(x, baseline):\n    # Write your implementation here.\n    raise NotImplementedError("Implement kernel")\n'
    return 'fn kernel(x: &Array2<i64>, baseline: &Array1<i64>) -> Array1<i64> {\n    // Write your implementation here.\n    todo!("Implement kernel")\n}\n'


def cases(task, hidden=False):
    # All practice cases are public: this is training, not a scored task.
    return [
        {'x': [[5, 1, 7], [2, 6, 3]], 'baseline': [3, 2, 4]},
        {'x': [[-3, 4, 1]], 'baseline': [-1, 4, -2]},
        {'x': [[-2], [3], [0]], 'baseline': [0]},
        {'x': [[1, 2], [3, 4]], 'baseline': [3, 4]},
    ]


def validate_input(task, data):
    if not isinstance(data, dict) or set(data) != {'x', 'baseline'}:
        raise ValueError('Input must contain exactly: x, baseline')
    arrays = []
    for name, rank in zip(names(task), RANKS[task]):
        arr = np.asarray(data[name])
        if arr.dtype.kind not in 'iu' or arr.ndim != rank or not all(arr.shape) or arr.size > 1024:
            raise ValueError(f'{name} must be a nonempty integer array with {rank} dimensions and at most 1,024 values')
        if np.any(arr < -512) or np.any(arr > 4096):
            raise ValueError('Values must be in [-512, 4096]')
        arrays.append(arr.astype(np.int64))
    if arrays[0].shape[1] != arrays[1].shape[0]:
        raise ValueError('baseline must contain one value per column of x')
    return arrays


def answer(task, data):
    x, baseline = validate_input(task, data)
    return [sum(max(int(v) - int(b), 0) for v, b in zip(row, baseline)) for row in x]


def task_payload(task, condition):
    return {
        'id': task, 'titles': TITLES[task], 'starter': starter(task, condition),
        'ranks': RANKS[task], 'output_rank': 1, 'arguments': names(task),
        'descriptions': ['x is a matrix with shape [N,C]. baseline is a vector with shape [C], containing one baseline value per column.\n\n1. Calculate d = x - baseline using broadcasting. d has shape [N,C].\n2. Replace each negative element of d with 0; keep other elements unchanged. Call the result a, with shape [N,C].\n3. Sum a along axis 1. Return the resulting vector with shape [N].\n\nExample: x=[[5,1,7],[2,6,3]] and baseline=[3,2,4] give a=[[2,0,3],[0,4,0]] and output [5,4].\n\nEquivalent pure-loop implementations are also accepted. Both practices are untimed and unscored. Pass all examples with your current code to continue.', 'x 是形状为 [N,C] 的矩阵。baseline 是形状为 [C] 的向量，为每列提供一个基准值。\n\n1. 利用广播计算 d = x - baseline，d 的形状为 [N,C]。\n2. 将 d 中每个负数改为 0，其余元素保持不变，得到形状为 [N,C] 的 a。\n3. 对 a 沿轴 1 求和，返回形状为 [N] 的向量。\n\n例如 x=[[5,1,7],[2,6,3]]，baseline=[3,2,4]，则 a=[[2,0,3],[0,4,0]]，输出为 [5,4]。\n\n也接受计算结果相同的纯循环实现。两次练习均不限时、不计分，用当前代码通过全部示例后即可继续。'],
        'requirements': [['d = x - baseline', 'Replace each negative element of d with 0', 'Sum a along axis 1', 'shape [N]'], ['d = x - baseline', '将 d 中每个负数改为 0', '对 a 沿轴 1 求和', '形状为 [N]']],
        'manual_topics': ['shape', 'broadcast', 'clamp', 'sum', 'allocate'],
        'examples': [{'name': f'Case {i+1}', 'input': data, 'expected': answer(task, data)} for i, data in enumerate(cases(task))],
    }
