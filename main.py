from zinnia import *
import numpy as np

@zk_circuit
def hello_world_circuit(a: int, b: int, c: Public[int]):
    assert a + b == c

if hello_world_circuit(10, 20, 30):
    print("Hello-World Circuit Constraints Fullfilled!")

if hello_world_circuit(1, 1, 3):
    print("Hello-World Circuit Constraints Not Fullfilled!")

foo_circuit = ZKCircuit.from_method(hello_world_circuit)
program = foo_circuit.compile()
print(program.source)
