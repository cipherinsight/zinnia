from zinnia import *


@zk_circuit
def verify_solution(input: NDArray[int, 5, 48], result: NDArray[int, 3, 46]):
    zero_rows = []
    zero_cols = []
    for i in range(5):
        zero_rows.append(all(input[i, :] == 0))
    for i in range(48):
        zero_cols.append(all(input[:, i] == 0))
    idx = 0
    flatten_result = result.flatten()
    for i in range(5):
        for j in range(48):
            if zero_rows[i] or zero_cols[j]:
                continue
            assert flatten_result[idx] == input[i, j]
            idx += 1
