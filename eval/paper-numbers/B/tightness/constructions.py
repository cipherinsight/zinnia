"""Method (a): inputs constructed by reading each program.

`CONSTRUCTIONS[program]` is a list of (label, overrides, reason). Each construction starts
from fixture input A and replaces the listed parameters, so parameters that do not affect
any size keep their fixture value. Which arrays a construction fills is not asserted here:
`run_search.py` measures every construction on every array with the NumPy reference.
"""
import numpy as np


def ramp(n, step=1.0, start=1.0):
    return [start + step * i for i in range(n)]


def full(shape, v):
    return np.full(shape, v).tolist()


def alternating(n, a=1, b=2):
    return [a if i % 2 == 0 else b for i in range(n)]


CONSTRUCTIONS = {
    'case0298': [
        ('distinct', {'a': ramp(5)}, '5 distinct values: np.unique has 5 entries, zeros is 5x5'),
    ],
    'case0301': [('ncol=4', {'ncol': 4}, 'a reshape of the fixed 20 elements: always 20')],
    'case0302': [('nrow=4', {'nrow': 4}, 'a reshape of the fixed 20 elements: always 20')],
    'case0303': [('ncol=1', {'ncol': 1}, 'col = (23 // 1) * 1 = 23: the whole array')],
    'case0304': [('ncol=1', {'ncol': 1}, 'col = 23: A[0:] is the whole array')],
    'case0322': [('all-equal', {'a': full((2, 2), 7)}, 'every element equals the minimum: 4 index pairs')],
    'case0353': [
        ('A-rows-not-in-B', {'A': [[1, 1, 1], [2, 2, 2]], 'B': full((7, 3), 0)},
         'no row of A occurs in B, so both rows are kept'),
    ],
    'case0383': [('N=100', {'N': 100}, 'N = len(a): all 100 indices')],
    'case0389': [('h=18,w=4', {'h': 18, 'w': 4}, 'a reshape of the fixed 72 elements: always 72')],
    'case0391': [('low=0,high=10', {'low': 0, 'high': 10}, 'the full column range')],
    'case0392': [('low=0,high=20', {'low': 0, 'high': 20}, 'the full row range')],
    'case0393': [('low=0,high=10', {'low': 0, 'high': 10}, 'the full column range')],
    'case0412': [('all-nonnegative', {'x': ramp(10)}, 'x >= 0 everywhere')],
    'case0413': [('all-imag-nonzero', {'x': [[float(i), 1.0] for i in range(10)]}, 'imaginary part 1 everywhere')],
    'case0414': [('bin_size=1', {'bin_size': 1}, '8 // 1 * 1 = 8 elements, 8 bins')],
    'case0415': [('bin_size=1', {'bin_size': 1}, '8 elements, 8 bins')],
    'case0416': [('bin_size=1', {'bin_size': 1}, '8x6 elements, 8x6 bins')],
    'case0417': [('bin_size=1', {'bin_size': 1}, '8 elements, 8 bins')],
    'case0418': [('bin_size=1', {'bin_size': 1}, '8x6 elements, 8x6 bins')],
    'case0419': [('bin_size=1', {'bin_size': 1}, '8x6 elements, 8x6 bins')],
    'case0442': [('A-disjoint-from-B', {'A': ramp(20, 1, 10), 'B': [1, 2, 3]}, 'no element of A is in B')],
    'case0443': [('A-inside-B', {'A': full(20, 2), 'B': [1, 2, 3]}, 'every element of A is in B')],
    'case0444': [('A-in-(B0,B1)', {'A': full(20, 5), 'B': [0, 10, 20]}, 'B0 < A < B1 everywhere')],
    'case0462': [('alternating-nonzero', {'a': alternating(20)}, 'no element is 0 or equal to its predecessor')],
    'case0463': [('alternating-nonzero', {'a': [[v] for v in alternating(20)]}, 'no element is 0 or equal to its predecessor')],
    'case0467': [('size=[8,8]', {'size': [8, 8]}, 'half-window 4 >= 3: every window is the whole 4x4 array')],
    'case0468': [('size=[8,8]', {'size': [8, 8]}, 'half-window 4 >= 3: every window is the whole 4x4 array')],
    'case0481': [('all-match', {'x': full(9, 3), 'y': full(9, 4), 'a': 3, 'b': 4}, 'x == a and y == b everywhere')],
    'case0490': [
        ('all-below-t1', {'arr': full((5, 50), 0.0), 'n1': full(5, 1), 'n2': full(5, 2)},
         'temp < t1 everywhere (fills np.where(temp < t1))'),
        ('all-at-least-t2', {'arr': full((5, 50), 0.0), 'n1': full(5, -2), 'n2': full(5, -1)},
         'temp >= t2 everywhere (fills np.where(temp >= t2))'),
        ('all-in-[t1,t2)', {'arr': full((5, 50), 0.0), 'n1': full(5, 0), 'n2': full(5, 1)},
         't1 <= temp < t2 everywhere (fills the mask result and its + 5)'),
        ('t2<t1', {'arr': full((5, 50), 0.0), 'n1': full(5, 1), 'n2': full(5, -1)},
         'temp < t1 and temp >= t2 everywhere: both np.where results full at once'),
    ],
    'case0507': [('all-nonzero', {'im': full((5, 6), 1)}, 'every row and column has a nonzero; crop is the whole image')],
    'case0508': [('all-nonzero', {'A': full((10, 10), 1)}, 'argwhere lists all 100 positions; crop is the whole array')],
    'case0509': [('all-zero', {'im': full((10, 10), 0.0)}, 'every row and column has a zero; crop is the whole image')],
    'case0510': [('all-nonzero', {'im': full((10, 10), 1.0)}, 'every row and column has a nonzero; crop is the whole image')],
    'pythran_periodic_dist': [
        ('L=-2,equal', {'x': full(16, 0.0), 'y': full(16, 0.0), 'z': full(16, 0.0), 'L': -2,
                        'periodicX': True, 'periodicY': True, 'periodicZ': True},
         'L < 0: every difference 0 exceeds L/2 = -1, so the first mask is all 256'),
        ('L=2,spread', {'x': ramp(16, 2.0, 0.0), 'y': ramp(16, 2.0, 0.0), 'z': ramp(16, 2.0, 0.0), 'L': 2,
                        'periodicX': True, 'periodicY': True, 'periodicZ': True},
         'L > 0, distinct points 2 apart: one of each off-diagonal pair is in each mask (120 of 256)'),
    ],
    'pythran_projection_simplex': [
        ('large-z', {'v': ramp(10, 1.0, 1.0), 'z': 1000},
         'z much larger than sum(v): u_k - (S_k - z)/k > 0 for every k'),
    ],
    'pythran_tax': [
        ('all>256303', {'d': full(10, 300000.0)}, 'every income in the top bracket'),
        ('all-in-(54057,256303]', {'d': full(10, 100000.0)}, 'every income in bracket 2'),
        ('all-in-(13769,54057]', {'d': full(10, 20000.0)}, 'every income in bracket 3 (6 arrays depend on it)'),
        ('all-in-(8820,13769]', {'d': full(10, 10000.0)}, 'every income in bracket 4'),
    ],
}
