from zinnia import *


@zk_circuit
def kmeans(
        data: NDArray[float, 10, 2],
        centroids: NDArray[float, 3, 2],
        classifications: NDArray[int, 10],
):
    n, d = data.shape
    classes = centroids.shape[0]
    labels = np.zeros((n, ), dtype=int)
    for _ in range(30):
        for i in range(n):
            dists = np.zeros((classes, ), dtype=float)
            for j in range(classes):
                dist = data[i] - centroids[j]
                dist = dist[0] * dist[0] + dist[1] * dist[1]
                dists[j] = dist
            labels[i] = np.argmin(dists)
        new_centroids = np.zeros((classes, d), dtype=float)
        counts = np.zeros((classes, ), dtype=int)
        for i in range(n):
            new_centroids[labels[i]] += data[i]
            counts[labels[i]] += 1
        for i in range(classes):
            if counts[i] == 0:
                continue
            new_centroids[i] /= counts[i]
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    assert labels == classifications



assert kmeans(
    np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4], [5, 5], [5, 6]]),
    np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
)