import json

import numpy as np
# a = np.array( # dims: 3x3x2
#     [[[ 0,  1],
#      [ 2,  3],
#      [ 4,  5]],
#     [[ 6,  7],
#      [ 8,  9],
#      [10, 11]],
#     [[12, 13],
#      [14, 15],
#      [16, 17]]]
# )
# b = np.array( # dims: 3x3
#     [[0, 1, 1],
#     [1, 0, 1],
#     [1, 1, 0]]
# )
# # select the elements in a according to b
# # to achieve this result:
# desired = np.array(
#   [[ 0,  3,  5],
#    [ 7,  8, 11],
#    [13, 15, 16]]
# )
#
# At first, I thought this must have a simple solution but I could not find one at all. Since I would like to port it to tensorflow, I would appreciate if somebody knows a numpy-type solution for this.
# A:
# <code>
# import numpy as np
# a = np.array(
#     [[[ 0,  1],
#      [ 2,  3],
#      [ 4,  5]],
#     [[ 6,  7],
#      [ 8,  9],
#      [10, 11]],
#     [[12, 13],
#      [14, 15],
#      [16, 17]]]
# )
# b = np.array(
#     [[0, 1, 1],
#     [1, 0, 1],
#     [1, 1, 0]]
# )
# </code>
# result = ... # put solution in this variable
# BEGIN SOLUTION
# <code>
#
# ------------------------------------------------------------
# result = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]

from zinnia import *

import numpy as np

@zk_circuit
def verify_solution(a: NDArray[int, 3, 3, 16], b: NDArray[int, 3, 3], desired: NDArray[int, 3, 3]):
    for i in range(3):
        for j in range(3):
            assert b[i, j] == 0 or b[i, j] == 1
            if b[i, j] == 0:
                assert a[i, j, 0] == desired[i, j]
            else:
                assert a[i, j, 1] == desired[i, j]
