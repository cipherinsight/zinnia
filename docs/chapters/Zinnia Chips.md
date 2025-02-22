# Zinnia Chips

Chips are another key concept in Zinnia. Chips acts like functions, providing a programming paradigm for you to separate the complex code into different components and allows you to write reusable code.

## Declaring Chips

Just like circuits, you can also turn your Python method into chips by simply annotating it with `@zk_chip` . Here is an example of `@zk_chip` :

```python
@zk_chip
def is_thirteen(x: int) -> None:
    assert x == 13

@zk_circuit
def my_circuit(x: int):
    if x != 0:
        is_thirteen(x)
```

This piece of code illustrates how to define a chip. This chip, `is_thirteen` , applies a constraint of `x` should be equal to 13 by using `assert` . This chip is then invoked by the circuit `my_circuit` . This circuit ensures that the input `x` should be either 0 or 13.

By using Zinnia, you can conveniently turn your Python methods into chips to be further called by circuits!

## Chip Parameters

Chips in Zinnia can also have parameters declared. As shown in the above example, there is a parameter `x` declared in the chip.

However, Zinnia simplifies the chip declaration for you. For chips, you can ignore the type annotation for parameters, as Zinnia is confident to infer them at the compile time! This makes the follow example possible:

```python
@zk_chip
def is_thirteen(x) -> None:
    assert x == 13

@zk_circuit
def my_circuit(x: int):
    if x != 0:
        is_thirteen(x)
    ary = np.asarray([x, x, x])
    is_thirteen(ary)
```

This is an extended example of our first chip example. In this example, you can see that the type annotation for `x` is omitted. Looking into the circuit, you can then find out that there are two different chip calls on `is_thirteen` . The first call, ensures that the number `x` should be 13. However in the second call, this ensures that every element in the array `ary` should be 13. 

Zinnia achieves that by inferring the type of each chip parameter at compile time case by case. As a result, chips works more like a “template” in many existing high level languages.

## Chip Returns

As we have mentioned before, the Zinnia chips works like functions or templates. Like functions in programming languages, Zinnia chips also support returning. 

```python
@zk_chip
def my_chip(x: int) -> int:
    if x < 10:
	      return x + 1
	  else:
	      return x - 1
```

In this example, you can find out that there is a chip of `return` statements. Zinnia ensures that your intended return value will be correctly receives by the chip caller, even if the branch condition is undetermined at compile time.

### The Return Existence

However, different from Python, Zinnia requires the chip to ensure that it can have a return value. Suppose that there is a chip defined as the follows. In this chip, the chip will have a return value when `x < 10`  but the return value does not exists when `x >= 10` . Zinnia will raise an error if such case is found during compilation.

```python
@zk_chip
def my_chip(x: int) -> int:
    if x < 10:
	      return x + 1
```

### The Return Annotation

As Zinnia is a ZKP circuit compiler, Zinnia need type information to assist it to know what computations and how many computations to generated for your code. As a result, it is crucial for Zinnia to know the return type for your declared chip. You are required to annotate the return type for each chip.

**Case 1: No Annotation / None Annotation**

The return type is usually required in Zinnia, however, it is allowed to declare chips without any annotations. In this case, the return type is interpreted as `None` . When the return type is `None` , you are not allowed to return any expression whose value is not `None`. You can still use `return` to terminate that chip.

```python
@zk_chip
def my_chip(x: int) -> None:
    for i in range(10):
        if i == x:
            return  # Allowed
        if i > x:
            return None  # Allowed
    return 0  # Error! Returning a value other that None is not allowed
```

**Case 2: Normal Annotation**

If you would like to return some value to the chip caller, annotate your chip with a return type is a good practice. You can only return expressions whose inferred datatype matches the return type you have annotated in the chip signature.

```python
@zk_chip
def find_value_indices(ary: NDArray[int, 10, 10], x: int) -> Tuple[int, int]:
    for i in range(10):
        for j in range(10):
            if ary[i][j] == x:
                return i, j  # Allowed
    return -1  # Error! Expr type (int) does not match return type (Tuple[int, int])
```