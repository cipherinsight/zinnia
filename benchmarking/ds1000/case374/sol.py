import json

from zinnia import *


@zk_circuit
def verify_solution(grades: NDArray[float, 27], eval: NDArray[float, 3], result: NDArray[float, 3]):
    # Goal: emulate R's ecdf(x)(x) applied to eval array.
    # Zinnia lacks sort, so we verify sortedness of grades.
    n = 27
    m = 3

    # 1) Verify grades is sorted in non-decreasing order
    for i in range(n - 1):
        assert grades[i] <= grades[i + 1]

    # 2) Build ECDF ys = i/n for i=1..n
    ys = np.zeros((n, ), dtype=int)
    for i in range(n):
        ys[i] = (i + 1) / float(n)

    # 3) Apply ECDF function to eval elements
    computed = np.zeros((m, ), dtype=int)
    for i in range(m):
        x = eval[i]
        if x < grades[0]:
            computed[i] = 0.0
        elif x >= grades[n - 1]:
            computed[i] = 1.0
        else:
            # Find smallest j such that grades[j] > x
            j = 0
            for k in range(n):
                if grades[k] > x:
                    j = k
                    break
            computed[i] = ys[j - 1]

    assert np.allclose(computed, result)


if __name__ == '__main__':
    grades = [
        60.8, 61.0, 65.5, 69.0, 76.0, 76.0, 78.0, 78.0, 82.0,
        86.0, 87.5, 89.5, 91.0, 91.5, 92.3, 92.5, 92.8, 93.0,
        93.5, 93.5, 94.5, 94.5, 95.0, 95.5, 98.0, 98.5, 99.5
    ]
    eval = [88.0, 87.0, 62.0]
    result = [11.0 / 27.0, 10.0 / 27.0, 2.0 / 27.0]

    assert verify_solution(grades, eval, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(grades, eval, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
