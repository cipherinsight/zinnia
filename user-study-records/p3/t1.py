from zinnia import *


@zk_chip
def is_prime(n: int) -> int:
    if n <= 1:
        return 0

    if n <= 3:
        return 1

    if n % 2 == 0 or n % 3 == 0:
        return 0

    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return 0
        i += 6

    return 1


@zk_circuit
def prime_test(n: Public[int], y: int):
    assert y == 0 or y == 1

    prime_result = is_prime(n)
    assert y == prime_result
