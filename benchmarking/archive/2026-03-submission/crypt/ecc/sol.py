from zinnia import *
import math

# P = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# def inv(x: int) -> int:
#     return pow(x, -1, P) % P


@zk_chip
def baby_check(x: int, y: int) -> None:
    a = 168700
    d = 168696
    x2 = x * x
    y2 = y * y
    assert a * x2 + y2 == 1 + d * x2 * y2


@zk_chip
def baby_add(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    a = 168700
    d = 168696
    beta = x1 * y2
    gamma = y1 * x2
    delta = (-a * x1 + y1) * (x2 + y2)
    tau = beta * gamma
    x = (beta + gamma) * math.inv(1 + d * tau)
    y = (delta + a * beta - gamma) * math.inv(1 - d * tau)
    return x, y


# baby_jubjub_ecc
@zk_circuit
def verify_solution(x1: int, y1: int, x2: int, y2: int, x3: int, y3: int):
    baby_check(x1, y1)
    baby_check(x2, y2)
    baby_check(x3, y3)
    x4, y4 = baby_add(x1, y1, x2, y2)
    assert x3 == x4
    assert y3 == y4

chips = [baby_check, baby_add]
# baby_jubjub_ecc(
#     995203441582195749578291179787384436505546430278305826713579947235728471134,
#     5472060717959818805561601436314318772137091100104008585924551046643952123905,
#     5299619240641551281634865583518297030282874472190772894086521144482721001553,
#     16950150798460657717958625567821834550301663161624707787222815936182638968203,
#     14805543388578810117460687107379140748822348273316260688573060998934016770136,
#     13589798946988221969763682225123791336245855044059976312385135587934609470572
# )
#
#
# print(ZKCircuit.from_method(verify_solution, chips=[baby_check, baby_add]).compile().source)