# Zinnia Docs

## Zinnia Introduction

Zinnia is a NEW transpiler in for Zero Knowledge Circuits. Zinnia lets you to write provable programs without requiring a deep understanding of the underlying ZK concepts. It aims to translate computations expressed in Python into computations on Zero Knowledge Circuits. Currently our primary focus is implementing its backend on halo-2. 

## Getting Started

### Installing Zinnia

Our project is hosted on Github (https://github.com/cipherinsight/zinnia). 

Please use `git clone` to clone our repository. You are advised to install all dependencies for Zinnia as listed below:

```bash
pip install pytest
pip install numpy
pip install z3 z3-solver
```

To verify your installation, you may execute `pytest` in the terminal. You may expect all unit-tests passed to indicate a successful installation.

### Writing Your First Zinnia Program

Now that we have Zinnia installed, let’s write our first Zinnia program! You can create an empty Python script named `main.py` (or other names you would prefer to choose).

Now, you may paste this code into your Python script:

```python
from zinnia import *
import numpy as np

@zk_circuit
def hello_world_circuit(a: int, b: int, c: Public[int]):
    assert a + b == c

if hello_world_circuit(10, 20, 30):
    print("Hello-World Circuit Constraints Fullfilled!")

if hello_world_circuit(1, 1, 3):
    print("Hello-World Circuit Constraints Not Fullfilled!")
```

You can execute this script with your Python Interpreter and you are expected to see those two lines printed on your screen. In the first case, `10 + 20 == 30` , so the constraints in this circuit have been fullfilled. In the second case, `1 + 1 != 3` , so the constraints in this circuit are NOT fullfilled.

Lets look into the source code. From line 4 to line 6, we defined a ZK circuit using Python grammar.

- We defined a method named `hello_world_circuit` decorated by `@zk_circuit` , indicating this method would be compiled as a Zero-Knowledge Circuit
- We specified all inputs for this circuit, including `a` , `b` and `c` . Notably, `c` is declared as a Public input and the other two inputs are private inputs by default.
- We added a constraint of `a + b == c` by using `assert`.

From line 8 to 12, we call that circuit by `hello_world_circuit` and specify all inputs to this circuit.

## What’s Next?

Congratulations on your first attempt on Zinnia! You may learn Zinnia in-depth now by our provided documents. Good luck with your learning journey!

[Zinnia Circuits](Zinnia%20Docs%2019f8a80edbed8073be20e332fe4ca2ec/Zinnia%20Circuits%2019f8a80edbed8087bc49c44a2e6bfc25.md)

[Zinnia Chips](Zinnia%20Docs%2019f8a80edbed8073be20e332fe4ca2ec/Zinnia%20Chips%2019f8a80edbed80e8ab10eeaf8cfd37e1.md)

[Control Flow](Zinnia%20Docs%2019f8a80edbed8073be20e332fe4ca2ec/Control%20Flow%2019f8a80edbed80bebae8e3520edecc6a.md)

[The Unknowns](Zinnia%20Docs%2019f8a80edbed8073be20e332fe4ca2ec/The%20Unknowns%2019f8a80edbed8084b8d0c0ebdb04e7f4.md)

[Array Manipulation](Zinnia%20Docs%2019f8a80edbed8073be20e332fe4ca2ec/Array%20Manipulation%2019f8a80edbed8099b275de24ca3d3097.md)

[Type Annotating](Zinnia%20Docs%2019f8a80edbed8073be20e332fe4ca2ec/Type%20Annotating%2019f8a80edbed8099b10cc99e72017f63.md)