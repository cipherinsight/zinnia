# The Unknowns

<aside>
🚧

Working in Progress

</aside>

In Zinnia, the Unknowns are those whose concrete value cannot be interpreted at compile time. 

## Two Core Restrictions in Zinnia

1. The datatype of each expression must be able to be inferred at compile time
    - The datatype of ndarray should be known
    - The shape of ndarray should be known upon creation
    - The start, stop, step of a range should be known
    - The start, stop, step of slice should be known
    - When repeating an array, the number of repetitions should be known
    - When conducting an axis-wise operation, the axis value should be known
    - …etc
2. The number of iterations in a loop or recursion must be able to be known at compile time

## Assigning identifiers in the outer scope

When assigning identifiers in the outer scope in Zinnia, there is a rule:

### Rule 1

It is generally not allowed to change the datatype of identifiers in the outer scope. 

```python
@zk_circuit
def foo(x: int):
    ary = np.asarray([1, 1, 1])
    if x == 4:
        ary = np.asarray([1, 1, 1, 1])  # Error! Change of the datatype of outer identifiers
```

In this example, the datatype of `ary` has been changed from `NDArray[int, 3]` into `NDArray[int, 4]`  which is not allowed.

### Rule 2

Under some cases, you can change the datatype of identifiers in the outer scope.

```python
@zk_circuit
def foo(x: int):
    ary = np.asarray([1, 1, 1])
    y = 5
    if y >= 4:
        ary = np.asarray([1, 1, 1, 1])  # Allowed
```

In this example, the change of datatype of `ary` is allowed. This is because the condition for that change, which is `y >= 4` , can be inferred at compile time. So this change will not incur side effects and the compiler thus allows that.

Note that, the commonly used `list.append` will also change the datatype of the list being appended. So the above two rules also applies to in-place functions like `list.append` .