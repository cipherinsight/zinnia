import json

from zinnia import *


@zk_circuit
def verify_solution(
        training_x: NDArray[float, 10, 2],
        training_y: NDArray[int, 10],
        testing_x: NDArray[float, 2, 2],
        testing_y: NDArray[int, 2]
):
    """
    Minimal depth-2 decision tree (two tests max):
      - Root stump: (feat_r, thr_r)
      - Left child stump: (feat_l, thr_l) used if x[feat_r] < thr_r
      - Right child stump: (feat_r2, thr_r2) used otherwise
    Each stump predicts: 1 if x[feat] >= thr else 0
    We search a SMALL, FIXED candidate set with deterministic tie-breaking.
    """

    # Fixed candidate sets (compile-time constants)
    thr = [0.5, 1.0, 1.5]  # thresholds
    n_thr = 3
    n_feat = 2
    n_train = 10
    n_test = 2

    # Best model holders
    best_err = 999
    best_feat_r = 0
    best_thr_r = 0
    best_feat_l = 0
    best_thr_l = 0
    best_feat_rr = 0
    best_thr_rr = 0

    # Enumerate all depth-2 trees with static nested loops
    for fr in range(n_feat):           # root feature
        for tr in range(n_thr):        # root threshold idx
            for fl in range(n_feat):   # left child feature
                for tl in range(n_thr):# left child thr idx
                    for fr2 in range(n_feat):  # right child feature
                        for tr2 in range(n_thr):  # right child thr idx
                            err = 0
                            # Evaluate training error
                            for i in range(n_train):
                                # Root decision
                                go_right = 1 if training_x[i, fr] >= thr[tr] else 0
                                pred = 0
                                if go_right == 0:
                                    # Left branch prediction
                                    pred = 1 if training_x[i, fl] >= thr[tl] else 0
                                else:
                                    # Right branch prediction
                                    pred = 1 if training_x[i, fr2] >= thr[tr2] else 0
                                if pred != training_y[i]:
                                    err += 1
                            # Argmin with deterministic tie-breaking on (fr,tr,fl,tl,fr2,tr2)
                            if err < best_err:
                                best_err = err
                                best_feat_r, best_thr_r = fr, tr
                                best_feat_l, best_thr_l = fl, tl
                                best_feat_rr, best_thr_rr = fr2, tr2
                            elif err == best_err:
                                if (fr < best_feat_r or
                                    (fr == best_feat_r and (tr < best_thr_r or
                                        (tr == best_thr_r and (fl < best_feat_l or
                                            (fl == best_feat_l and (tl < best_thr_l or
                                                (tl == best_thr_l and (fr2 < best_feat_rr or
                                                    (fr2 == best_feat_rr and tr2 < best_thr_rr)
                                                ))
                                            ))
                                        ))
                                    ))):
                                    best_feat_r, best_thr_r = fr, tr
                                    best_feat_l, best_thr_l = fl, tl
                                    best_feat_rr, best_thr_rr = fr2, tr2

    # Evaluate on the test set
    test_errors = 0
    for i in range(n_test):
        go_right = 1 if testing_x[i, best_feat_r] >= thr[best_thr_r] else 0
        pred = 0
        if go_right == 0:
            pred = 1 if testing_x[i, best_feat_l] >= thr[best_thr_l] else 0
        else:
            pred = 1 if testing_x[i, best_feat_rr] >= thr[best_thr_rr] else 0
        if pred != testing_y[i]:
            test_errors += 1

    # For this tiny instance, allow at most 1 test error
    assert test_errors <= 1


if __name__ == '__main__':
    # Construct a tiny dataset separable by a depth-2 rule:
    # Root on feature 0 with thr ≈ 1.0, then refine the right branch by feature 1 with thr ≈ 0.5.
    training_x = [
        [0.2, 0.1],   # left  -> y=0
        [0.5, -1.0],  # left  -> y=0
        [1.2, 0.0],   # right -> x1<0.5 -> y=0
        [1.8, -0.3],  # right -> x1<0.5 -> y=0
        [2.2, 1.0],   # right -> x1>=0.5 -> y=1
        [0.7, 0.5],   # left  -> y=0
        [1.1, -0.2],  # right -> x1<0.5 -> y=0
        [2.8, 0.9],   # right -> x1>=0.5 -> y=1
        [0.3, -0.4],  # left  -> y=0
        [1.6, 1.2],   # right -> x1>=0.5 -> y=1
    ]
    training_y = [0, 0, 0, 0, 1, 0, 0, 1, 0, 1]

    testing_x = [
        [1.4, 0.0],  # right -> x1<0.5 -> 0
        [0.4, 2.0],  # left  -> 0
    ]
    testing_y = [0, 0]

    assert verify_solution(training_x, training_y, testing_x, testing_y)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(training_x, training_y, testing_x, testing_y)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
