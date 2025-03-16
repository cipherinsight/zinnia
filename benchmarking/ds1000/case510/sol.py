# No. 510
# Problem:
# I want to process a gray image in the form of np.array.
# *EDIT: chose a slightly more complex example to clarify
# Suppose:
# im = np.array([ [0,0,0,0,0,0] [0,0,5,1,2,0] [0,1,8,0,1,0] [0,0,0,7,1,0] [0,0,0,0,0,0]])
# I'm trying to create this:
# [ [0,5,1,2], [1,8,0,1], [0,0,7,1] ]
# That is, to remove the peripheral zeros(black pixels) that fill an entire row/column.
# In extreme cases, an image can be totally black, and I want the result to be an empty array.
# I can brute force this with loops, but intuitively I feel like numpy has a better means of doing this.
# A:
# <code>
# import numpy as np
# im = np.array([[0,0,0,0,0,0],
#                [0,0,5,1,2,0],
#                [0,1,8,0,1,0],
#                [0,0,0,7,1,0],
#                [0,0,0,0,0,0]])
# </code>
# result = ... # put solution in this variable
# BEGIN SOLUTION
# <code>
#
# ------------------------------------------------------------
# mask = im == 0
# rows = np.flatnonzero((~mask).sum(axis=1))
# cols = np.flatnonzero((~mask).sum(axis=0))
# if rows.shape[0] == 0:
#     result = np.array([])
# else:
#     result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
import json

from zinnia import *

import numpy as np

from zinnia.config import OptimizationConfig

im = np.array([[0,0,0,0,0,0],
               [0,0,5,1,2,0],
               [0,1,8,0,1,0],
               [0,0,0,7,1,0],
               [0,0,0,0,0,0]])
mask = im == 0
rows = np.flatnonzero((~mask).sum(axis=1))
cols = np.flatnonzero((~mask).sum(axis=0))
if rows.shape[0] == 0:
    result = np.array([])
else:
    result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]


@zk_circuit
def verify_solution(input: NDArray[int, 5, 6], result: NDArray[int, 3, 4]):
    zero_rows = []
    zero_cols = []
    for i in range(5):
        zero_rows.append(all(input[i, :] == 0))
    for i in range(6):
        zero_cols.append(all(input[:, i] == 0))
    idx = 0
    flatten_result = result.flatten()
    for i in range(5):
        for j in range(6):
            if zero_rows[i] or zero_cols[j]:
                continue
            assert flatten_result[idx] == input[i, j]
            idx += 1

# print(result)
# assert verify_solution(im, result)

# Parse inputs
# program = ZKCircuit.from_method(verify_solution, config=ZinniaConfig(optimization_config=OptimizationConfig(
#     always_satisfied_elimination=False,
#     constant_fold=False,
#     dead_code_elimination=False,
#     duplicate_code_elimination=False,
#     shortcut_optimization=False,
# ))).compile()
# print(program.source)
# parsed_inputs = program.argparse(im, result)
# json_dict = {}
# for entry in parsed_inputs.entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
