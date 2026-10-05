import json

from zinnia import *


@zk_circuit
def verify_solution(
        data: NDArray[float, 10, 2],
        centroids: NDArray[float, 3, 2],
        classifications: NDArray[int, 10],
):
    n, d = data.shape
    classes = centroids.shape[0]
    labels = np.zeros((n, ), dtype=int)
    for _ in range(10):
        for i in range(n):
            dists = np.zeros((classes, ), dtype=float)
            for j in range(classes):
                dist = data[i] - centroids[j]
                dist = dist[0] * dist[0] + dist[1] * dist[1]
                dists[j] = dist
            labels[i] = np.argmin(dists)
        new_centroids = np.zeros((classes, d), dtype=float)
        counts = np.zeros((classes, ), dtype=float)
        for i in range(n):
            new_centroids[labels[i]] += data[i]
            counts[labels[i]] += 1.0
        for i in range(classes):
            new_centroids[i] /= counts[i]
        centroids = new_centroids
    assert labels == classifications



# assert verify_solution(
#     np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4], [5, 5], [5, 6]]),
#     np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
#     [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
# )

# entries = ZKCircuit.from_method(verify_solution).argparse(
#     np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4], [5, 5], [5, 6]]),
#     np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
#     [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
# ).entries
#
# json_dict = {}
# for entry in entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))

# json_dict = {}
# json_dict["data"] = [float(x) for x in np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4], [5, 5], [5, 6]]).flatten().tolist()]
# json_dict["centroids"] = [float(x) for x in np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]).flatten().tolist()]
# json_dict["classifications"] = [int(x) for x in [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]]
# print(json.dumps(json_dict, indent=2))

# with open("compiled.rs", "w") as f:
#     f.write(ZKCircuit.from_method(verify_solution).compile().source)

# 327313 cells rust version
# 427421 cells python version