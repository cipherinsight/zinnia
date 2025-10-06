import json

from zinnia import *


@zk_circuit
def verify_solution(A: NDArray[int, 4, 3], B: NDArray[int, 7, 3], output: NDArray[int, 2, 3]):
    # Step 1: For each row i of A, check membership in B (exact row match on 3 columns)
    in_B = [False, False, False, False]

    for i in range(4):
        found = False
        for j in range(7):
            m0 = (A[i, 0] == B[j, 0])
            m1 = (A[i, 1] == B[j, 1])
            m2 = (A[i, 2] == B[j, 2])
            row_match = m0 and m1 and m2
            if row_match:
                found = True
        in_B[i] = found

    # Step 2: prefix counts of rows NOT in B, to determine positions in the kept list
    pref = 0                                   # number of kept rows seen so far
    pref_before = [0, 0, 0, 0]
    keep_flag   = [0, 0, 0, 0]                 # 1 if A[i] not in B else 0

    for i in range(4):
        pref_before[i] = pref
        not_in = 0 if in_B[i] else 1
        keep_flag[i] = not_in
        pref = pref + not_in

    # Exactly two rows should be kept for this instance
    assert pref == 2

    # Step 3: Build expected kept rows using indicators:
    # If keep_flag[i]==1 and pref_before[i]==0 -> goes to kept row 0
    # If keep_flag[i]==1 and pref_before[i]==1 -> goes to kept row 1
    exp = [
        [0, 0, 0],
        [0, 0, 0],
    ]
    for i in range(4):
        is_keep = keep_flag[i]

        is_pos0 = 1 if pref_before[i] == 0 else 0
        is_pos1 = 1 if pref_before[i] == 1 else 0

        w0 = is_keep * is_pos0
        w1 = is_keep * is_pos1

        for c in range(3):
            exp[0][c] = exp[0][c] + A[i, c] * w0
            exp[1][c] = exp[1][c] + A[i, c] * w1

    # Step 4: Compare with provided output
    for r in range(2):
        for c in range(3):
            assert output[r, c] == exp[r][c]



if __name__ == '__main__':
    A = [
        [1, 1, 1],
        [1, 1, 2],
        [1, 1, 3],
        [1, 1, 4],
    ]
    B = [
        [0, 0, 0],
        [1, 0, 2],
        [1, 0, 3],
        [1, 0, 4],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 4],
    ]
    output = [
        [1, 1, 2],
        [1, 1, 3],
    ]
    assert verify_solution(A, B, output)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(A, B, output)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
