# Source: NPBench polybench/floyd_warshall (floyd_warshall_numpy.py)
# Original signature: kernel(path) where path is NxN int (np.int32).
# Migration notes:
#   - N hoisted as module-level constant.
#   - path.shape[0] replaced with N for static loop bound.
#   - np.add.outer is likely unsupported but kept verbatim.
from zinnia import *

N = 200


@zk_circuit
def floyd_warshall(path: NDArray[Integer, 200, 200]):
    for k in range(N):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
