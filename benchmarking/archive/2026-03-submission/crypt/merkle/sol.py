import json
import math

from zinnia import *

# MiMC-3 hash reused from earlier example
@zk_chip
def mimc3(x: int, k: int) -> int:
    # 8 fixed constants for rounds
    c = [1, 2, 3, 4, 5, 6, 7, 8]
    t = x + k
    for i in range(8):
        t = (t + c[i]) * (t + c[i]) * (t + c[i])
    return t


@zk_chip
def mimc_hash2(left: int, right: int) -> int:
    # Compress 2 field elements into 1 hash
    return mimc3(left + right, 0)


# --------------------------------------------------------------------
# Build a fixed-depth Merkle tree (depth = 3 in this demo)
# Leaves: 8 elements. Tree has 3 layers of hashing.

@zk_chip
def merkle_root(leaves: NDArray[int, 8]) -> int:
    # Level 0 → Level 1
    L1 = []
    for i in range(0, 8, 2):
        L1.append(mimc_hash2(leaves[i], leaves[i + 1]))

    # Level 1 → Level 2
    L2 = []
    for i in range(0, 4, 2):
        L2.append(mimc_hash2(L1[i], L1[i + 1]))

    # Level 2 → Root
    root = mimc_hash2(L2[0], L2[1])
    return root


# --------------------------------------------------------------------
# Verify an authentication path of fixed depth (3)
# For leaf index in [0,7], prove its inclusion in root.

@zk_chip
def merkle_verify(leaf: int, path: NDArray[int, 3], index_bits: NDArray[int, 3], root: int) -> None:
    cur = leaf
    for d in range(3):
        if index_bits[d] == 0:
            cur = mimc_hash2(cur, path[d])
        else:
            cur = mimc_hash2(path[d], cur)
    assert cur == root


# --------------------------------------------------------------------
@zk_circuit
def verify_solution(leaves: NDArray[int, 8], leaf_idx: int, path: NDArray[int, 3], bits: NDArray[int, 3]):
    root = merkle_root(leaves)
    leaf = leaves[leaf_idx]
    merkle_verify(leaf, path, bits, root)


chips = [mimc3, mimc_hash2, merkle_root, merkle_verify]


if __name__ == '__main__':
    leaves = [11, 22, 33, 44, 55, 66, 77, 88]
    leaf_idx = 5
    bits = [1, 0, 1]
    path = [55, 11523261815481333160108258185116327645151901011750749413828919435813935579027,
            21212311445138905926880864250103604663610259329539511211264150430276934290235]

    # Leaf index: 5
    # Merkle Root: 12642995768575162563634759873535157214939979179441040718598881362661940465302
    # Authentication Path: [55, 11523261815481333160108258185116327645151901011750749413828919435813935579027, 21212311445138905926880864250103604663610259329539511211264150430276934290235]
    # Index bits: [1, 0, 1]

    # verify_solution(leaves, leaf_idx, path, bits)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution, chips=chips).compile()
    # parsed_inputs = program.argparse(msg, expected)
    # json_dict = {}
    # for entry in parsed_inputs.entries:
    #     json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump({
            "x_0_0_0": str(leaves[0]),
            "x_0_0_1": str(leaves[1]),
            "x_0_0_2": str(leaves[2]),
            "x_0_0_3": str(leaves[3]),
            "x_0_0_4": str(leaves[4]),
            "x_0_0_5": str(leaves[5]),
            "x_0_0_6": str(leaves[6]),
            "x_0_0_7": str(leaves[7]),
            "x_0_1": str(leaf_idx),
            "x_0_2_0": str(path[0]),
            "x_0_2_1": str(path[1]),
            "x_0_2_2": str(path[2]),
            "x_0_3_0": str(bits[0]),
            "x_0_3_1": str(bits[1]),
            "x_0_3_2": str(bits[2]),
        }, f, indent=2)
