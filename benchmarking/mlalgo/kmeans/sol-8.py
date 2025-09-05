import json

from zinnia import *


@zk_circuit
def verify_solution(
        data: NDArray[float, 80, 2],
        centroids: NDArray[float, 3, 2],
        classifications: NDArray[int, 80],
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

