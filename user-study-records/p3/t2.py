from zinnia import *


@zk_chip
def calculate_ways(n: int) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2

    ways = [0] * (50 + 1)
    ways[1] = 1  # There's 1 way to climb 1 step
    ways[2] = 2  # There are 2 ways to climb 2 steps

    for i in range(3, 50 + 1):
        ways[i] = ways[i - 1] + ways[i - 2]

    return ways[n]


@zk_circuit
def climbing_stairs(n: Public[int], y: int):
    assert 1 <= n and n <= 50

    correct_ways = calculate_ways(n)

    assert y == correct_ways
