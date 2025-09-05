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
def verify_solution(a: NDArray[int, 16, 4], result: NDArray[int, 16, 2, 2]):
    assert a.reshape((a.shape[0]//2, 2, a.shape[1]//2, 2)).transpose((0, 2, 1, 3)).reshape((16, 2, 2)) == result
