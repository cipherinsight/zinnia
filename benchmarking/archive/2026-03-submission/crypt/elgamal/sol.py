import json
import math

from zinnia import *

# MiMC-3 over the BN254 field (implicit mod p in the circuit arithmetic)
# Round function: x <- (x + c[r])^3  for r = 0..7
# Hash of fixed-length message: absorb 3 words with a single-rate sponge:
#   state = 0
#   for i in 0..2: state = MiMC_permute(state + msg[i])
#   output = state

# ElGamal public-key encryption scheme over BN254 (implicit mod p)
# Uses Zinnia's native field exponentiation (**), supporting up to 251-bit exponents.

@zk_chip
def elgamal_keygen(g: int, sk: int) -> int:
    # pk = g^sk mod p
    return g ** sk


@zk_chip
def elgamal_encrypt(g: int, pk: int, msg: int, r: int) -> Tuple[int, int]:
    # c1 = g^r
    # c2 = msg * pk^r
    c1 = g ** r
    c2 = msg * (pk ** r)
    return c1, c2


@zk_chip
def elgamal_decrypt(sk: int, c1: int, c2: int) -> int:
    # shared = c1^sk
    # msg = c2 * inv(shared)
    shared = c1 ** sk
    msg = c2 * math.inv(shared)
    return msg


@zk_circuit
def verify_solution(g: int, sk: int, r: int, msg: int):
    # Key generation
    pk = elgamal_keygen(g, sk)
    # Encryption
    c1, c2 = elgamal_encrypt(g, pk, msg, r)
    # Decryption
    recovered = elgamal_decrypt(sk, c1, c2)
    # Round-trip consistency check
    assert recovered == msg


chips = [elgamal_decrypt, elgamal_encrypt, elgamal_keygen]


if __name__ == '__main__':
    g = 5
    sk = 123456789123456789123456789123456789  # 128-bit scale
    r = 987654321987654321987654321987654321  # 128-bit scale
    msg = 42424242424242424242

    # assert verify_solution(msg, expected)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution, chips=chips).compile()
    # parsed_inputs = program.argparse(msg, expected)
    # json_dict = {}
    # for entry in parsed_inputs.entries:
    #     json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump({
            "x_0_0": str(g),
            "x_0_1": str(sk),
            "x_0_2": str(r),
            "x_0_3": str(msg)
        }, f, indent=2)
