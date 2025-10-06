import json

from zinnia import *

@zk_circuit
def verify_solution(A: NDArray[int, 4, 3], B: NDArray[int, 7, 3], output: NDArray[int, 7, 3]):
    # --- Step 1: membership flags ---
    # inB[i] = True iff A[i] appears as a row in B
    inB = [False, False, False, False]
    for i in range(4):
        found = False
        for j in range(7):
            m0 = (A[i, 0] == B[j, 0])
            m1 = (A[i, 1] == B[j, 1])
            m2 = (A[i, 2] == B[j, 2])
            if m0 and m1 and m2:
                found = True
        inB[i] = found

    # inA[j] = True iff B[j] appears as a row in A
    inA = [False] * 7
    for j in range(7):
        found = False
        for i in range(4):
            m0 = (B[j, 0] == A[i, 0])
            m1 = (B[j, 1] == A[i, 1])
            m2 = (B[j, 2] == A[i, 2])
            if m0 and m1 and m2:
                found = True
        inA[j] = found

    # --- Step 2: prefix counts for A-only and B-only ---
    # keep_A[i] = 1 if A[i] not in B else 0
    keep_A = [0, 0, 0, 0]
    prefA_before = [0, 0, 0, 0]
    prefA = 0
    for i in range(4):
        prefA_before[i] = prefA
        not_inB = 0 if inB[i] else 1
        keep_A[i] = not_inB
        prefA = prefA + not_inB
    # Exactly two rows A-only for this instance
    assert prefA == 2

    # keep_B[j] = 1 if B[j] not in A else 0
    keep_B = [0] * 7
    prefB_before = [0] * 7
    prefB = 0
    for j in range(7):
        prefB_before[j] = prefB
        not_inA = 0 if inA[j] else 1
        keep_B[j] = not_inA
        prefB = prefB + not_inA
    # Exactly five rows B-only for this instance
    assert prefB == 5

    # --- Step 3: construct expected symmetric difference ---
    # First two rows: A-only in A's order
    exp = [
        [0, 0, 0],  # row 0
        [0, 0, 0],  # row 1
        [0, 0, 0],  # row 2
        [0, 0, 0],  # row 3
        [0, 0, 0],  # row 4
        [0, 0, 0],  # row 5
        [0, 0, 0],  # row 6
    ]
    for i in range(4):
        is_keep = keep_A[i]
        at_pos0 = 1 if prefA_before[i] == 0 else 0
        at_pos1 = 1 if prefA_before[i] == 1 else 0
        w0 = is_keep * at_pos0  # goes to exp[0]
        w1 = is_keep * at_pos1  # goes to exp[1]
        for c in range(3):
            exp[0][c] = exp[0][c] + A[i, c] * w0
            exp[1][c] = exp[1][c] + A[i, c] * w1

    # Next five rows: B-only in B's order, placed at exp[2..7)
    for j in range(7):
        is_keep = keep_B[j]
        # position r in {0..4} -> absolute row = 2 + r
        for r in range(5):
            at_r = 1 if prefB_before[j] == r else 0
            w = is_keep * at_r
            for c in range(3):
                exp[2 + r][c] = exp[2 + r][c] + B[j, c] * w

    # --- Step 4: compare ---
    for r in range(7):
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
        [0, 0, 0],
        [1, 0, 2],
        [1, 0, 3],
        [1, 0, 4],
        [1, 1, 0],
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
