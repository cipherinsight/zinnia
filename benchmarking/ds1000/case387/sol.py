# No. 387
# Problem:
# I have a 2-d numpy array as follows:
# a = np.array([[1,5,9,13],
#               [2,6,10,14],
#               [3,7,11,15],
#               [4,8,12,16]]
# I want to extract it into patches of 2 by 2 sizes with out repeating the elements.
# The answer should exactly be the same. This can be 3-d array or list with the same order of elements as below:
# [[[1,5],
#  [2,6]],
#  [[9,13],
#  [10,14]],
#  [[3,7],
#  [4,8]],
#  [[11,15],
#  [12,16]]]
# How can do it easily?
# In my real problem the size of a is (36, 72). I can not do it one by one. I want programmatic way of doing it.
# A:
# <code>
# import numpy as np
# a = np.array([[1,5,9,13],
#               [2,6,10,14],
#               [3,7,11,15],
#               [4,8,12,16]])
# </code>
# result = ... # put solution in this variable
# BEGIN SOLUTION
# <code>
#
# ------------------------------------------------------------
# result = a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2).swapaxes(1, 2).reshape(-1, 2, 2)

# solution with transpose
# result = a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2).transpose(0, 2, 1, 3).reshape(-1, 2, 2)
import json

# result = [[[1,5],
# [2,6]],
# [[9,13],
# [10,14]],
# [[3,7],
# [4,8]],
# [[11,15],
# [12,16]]]


from zinnia import *

@zk_circuit
def verify_solution(a: DynamicNDArray[int, 16, 2], result: DynamicNDArray[int, 16, 3]):
    rows = a.shape[0] // 4
    cols = a.shape[0] // rows

    row_pairs = rows // 2
    col_pairs = cols // 2

    expected = a.reshape((row_pairs, 2, col_pairs, 2)).transpose((0, 2, 1, 3)).reshape((result.shape[0],))
    assert result.reshape((result.shape[0],)) == expected


# assert verify_solution(a, result)

# Parse inputs
# program = ZKCircuit.from_method(verify_solution).compile()
# parsed_inputs = program.argparse(a, result)
# json_dict = {}
# for entry in parsed_inputs.entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
