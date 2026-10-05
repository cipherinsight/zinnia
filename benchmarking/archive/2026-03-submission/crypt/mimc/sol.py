import json

from zinnia import *

# MiMC-3 over the BN254 field (implicit mod p in the circuit arithmetic)
# Round function: x <- (x + c[r])^3  for r = 0..7
# Hash of fixed-length message: absorb 3 words with a single-rate sponge:
#   state = 0
#   for i in 0..2: state = MiMC_permute(state + msg[i])
#   output = state

@zk_chip
def mimc_permute(x: int) -> int:
    # 8 small round constants (example values; fixed at compile time)
    c0, c1, c2, c3, c4, c5, c6, c7 = 1, 2, 3, 4, 5, 6, 7, 8

    # 8 rounds (statically bounded)
    x = (x + c0) * (x + c0) * (x + c0)
    x = (x + c1) * (x + c1) * (x + c1)
    x = (x + c2) * (x + c2) * (x + c2)
    x = (x + c3) * (x + c3) * (x + c3)
    x = (x + c4) * (x + c4) * (x + c4)
    x = (x + c5) * (x + c5) * (x + c5)
    x = (x + c6) * (x + c6) * (x + c6)
    x = (x + c7) * (x + c7) * (x + c7)
    return x


@zk_chip
def mimc3_hash_3(msg: NDArray[int, 3]) -> int:
    state = 0
    # absorb 3 words (fixed length = 3)
    state = mimc_permute(state + msg[0])
    state = mimc_permute(state + msg[1])
    state = mimc_permute(state + msg[2])
    return state  # 1-word digest


@zk_circuit
def verify_solution(msg: NDArray[int, 3], expected: int):
    # compute and check
    h = mimc3_hash_3(msg)
    assert h == expected


chips = [mimc_permute, mimc3_hash_3]


if __name__ == '__main__':
    msg = [1, 2, 3]
    expected = 13282693387779170360280659014090582903649482011954396102989514311726011132212

    # assert verify_solution(msg, expected)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution, chips=chips).compile()
    # parsed_inputs = program.argparse(msg, expected)
    # json_dict = {}
    # for entry in parsed_inputs.entries:
    #     json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump({
            "x_0_0_0": str(msg[0]),
            "x_0_0_1": str(msg[1]),
            "x_0_0_2": str(msg[2]),
            "x_0_1": str(expected)
        }, f, indent=2)
