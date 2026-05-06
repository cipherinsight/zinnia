# Source: Pythran tests/cases/mandel.py
# Original #pythran export: mandel(int, float, float, int)
from zinnia import *


@zk_circuit
def mandel(size: int, x_center: float, y_center: float, max_iteration: int):
    out = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            x, y = (x_center + 4.0 * float(i - size / 2) / size,
                    y_center + 4.0 * float(j - size / 2) / size)

            a, b = (0.0, 0.0)
            iteration = 0

            while (a ** 2 + b ** 2 <= 4.0 and iteration < max_iteration):
                a, b = a ** 2 - b ** 2 + x, 2 * a * b + y
                iteration += 1
            if iteration == max_iteration:
                color_value = 255
            else:
                color_value = iteration * 10 % 255
            out[i][j] = color_value
    _zinnia_result = out
