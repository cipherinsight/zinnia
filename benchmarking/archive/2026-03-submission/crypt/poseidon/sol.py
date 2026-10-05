import json

from zinnia import *


@zk_circuit
def verify_solution(dat: PoseidonHashed[NDArray[int, 10]]):
    assert sum(dat) == 55



# circuit = ZKCircuit.from_method(verify_solution)
# entries = circuit.argparse(
#     PoseidonHashed(
#         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "21888242871839275222246405745257275088548364400416034343698204186575808495617",
#     )
# ).entries
# data_dict = {}
# for entry in entries:
#     data_dict[entry.get_key()] = entry.get_value()
#
# print(json.dumps(data_dict, indent=2))