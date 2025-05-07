from zinnia import *


@zk_circuit
def fibonacci(n: int, result: int):
    assert n <= 100
    # fibonacci for n less than 100
    if n == 0:
        assert result == 0
    elif n == 1:
        assert result == 1
    else:
        a, b = 0, 1
        for i in range(1, 101):
            a, b = b, a + b
            if i == n:
                break
        assert a == result


assert fibonacci(10, 55)
# fib_circuit = ZKCircuit.from_method(fibonacci)
# program = fib_circuit.compile()
# print(program.source)