from zinnia import zk_circuit, NDArray


@zk_circuit
def find_nearest_point(points: NDArray[int, 10, 2], target: NDArray[int, 2], result: int):
    min_dist = (points[0, 0] - target[0]) ** 2 + (points[0, 1] - target[1]) ** 2
    min_idx = 0
    for i in range(1, 10):
        dist = (points[i, 0] - target[0]) ** 2 + (points[i, 1] - target[1]) ** 2
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    assert min_idx == result


assert find_nearest_point(
    [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]],
    [7, 7],
    6
)