# pythran program mandel.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda size, x_center, y_center, max_iteration: 0 <= size <= 1024)
@requires(lambda size, x_center, y_center, max_iteration: size != 0)
def mandel(size: Integer, x_center: Float, y_center: Float, max_iteration: Integer):
    out = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        'omp parallel for'
        for j in range(size):
            x, y = (x_center + 4.0 * float(i - size / 2) / size, y_center + 4.0 * float(j - size / 2) / size)
            a, b = (0.0, 0.0)
            iteration = 0
            while a ** 2 + b ** 2 <= 4.0 and iteration < max_iteration:
                a, b = (a ** 2 - b ** 2 + x, 2 * a * b + y)
                iteration += 1
            if iteration == max_iteration:
                color_value = 255
            else:
                color_value = iteration * 10 % 255
            out[i][j] = color_value
    return out
