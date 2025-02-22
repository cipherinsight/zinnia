# Zinnia Circuits

Circuits is a key concept in Zinnia. Each Zinnia circuit represents a unique Zero-Knowledge Program that can be further compiled into arithmetic circuits.

## Writing Circuits

As Zinnia is a Python-like language, we can write Zinnia circuits just like writing Python code. By annotating your Python method by `@zk_circuit` provided by Zinnia, you can instantly turn your Python method into a ZKP circuit within a few seconds!

Here is an example of annotating a Python method annotated by the `@zk_circuit` . Originally, it was a method testing the equality of two integer numbers `x` and `y` . By annotating it with `@zk_circuit` , we instantly turned it into a ZKP circuit which constraints the equality of two private inputs `x` and `y` . That was fascinating right?

```python
@zk_circuit
def my_circuit(x: int, y: int):
    assert x == y
```

By using Zinnia, you can turn a wide range of Python methods into ZKP circuits just with our annotator. However some restrictions may apply. 

## Circuit Syntax: The Inputs

As we have already know in the Zero Knowledge Concepts Tutorial, ZKP circuits are those who have inputs and constraints. Consequently, we should clearly declare the inputs for this circuit.

In Zinnia, you can declare circuit inputs through method parameters. Each parameter is a circuit input. In our previous example, two inputs are declared through the method parameters, namely `x` and `y` . 

In Zinnia, it is required to declare your input parameters together with type annotations. AS Python is a weak type language, however, Zinnia need the information to know how many computations and what computations should be done at compile time to correctly compile your program into a ZKP circuit. Thus, it is a requirement to clearly specify the datatype by annotation for each input parameter. 

For more details in writing type annotations, please read the type annotating chapter.

However, note that Zinnia only support positional args with no default values. We may consider to support this feature in the near future!

## Circuit Syntax: The Outputs

As we have already know in the Zero Knowledge Concepts Tutorial, ZKP circuits are those who does not have outputs. Actually, the outputs are treated as inputs to be passed into the circuit and the circuit then verifies the correctness of that output.

Zinnia follows this universal ZKP circuit concept, as a result, there is no return statement allowed inside a circuit. In addition to that, the return type annotation for the circuit method is also not allowed.

## Express Constraints in a Circuit

ZKP circuits consist of a set of constraints. So, the primary goal for Zinnia is converting a regular program into constraints. There are two kinds of constraints in Zinnia:

### Implicit Constraints

Implicit Constraints are those computations being expressed as constraints. Suppose that we have this computation illustrated as the follows:

```python
@zk_circuit
def foo(x: int, y: int):
    z = x * y
```

In this circuit, we computes `x * y` and assign it to `z` . In other traditional ZKP circuits, you are expected to write constraints to ensure that z is equal to `x * y` . However, Zinnia, implicitly adds that constraint for you.

### Explicit Constraints

Explicit Constraints are those constraints written by users explicitly through `assert` statements. Suppose that you have calculated the value to z outside the ZK circuit. How to enforce that z should be equal to `x * y` ? In this case, you may find the explicit constraints useful. Using `assert` statements, you can express constraints explicitly.

```python
@zk_circuit
def foo(x: int, y: int, z: int):
    assert z == x * y
```

As illustrated in this example, the equality between z and `x * y` has been enforced by explicit constraint with `assert`.