# Array Manipulation

<aside>
🚧

Working in Progress

</aside>

In Zinnia, you can manipulate arrays like Python. In addition to that, Zinnia added numpy support (partial) to help you manipulate complex arrays.

## List and Tuples

Zinnia supports lists and tuples in Python. They can act as a simple array

```python
@zk_circuit
def foo(x: int):
    a = [0, 0, 0, 0]
    a[x] = 1
    assert sum(a) == 1
```

## NDArray

Zinnia also supports passing and using NDArray. You may use the same syntax like numpy

```python
@zk_circuit
def foo():
    ary = np.arange(0, 10, 2, float)
    assert ary.tolist() == list(float(x) for x in range(0, 10, 2))

```

```python
@zk_circuit
def foo():
    array = np.asarray([[1, 2, 3], [4, 5, 6]])
    assert array.argmax(axis=0).tolist() == [1, 1, 1]
    assert array.argmax(axis=1).tolist() == [2, 2]

```

```python
@zk_circuit
def foo():
    array = np.asarray([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    assert array[0, 0] == array[0][0] == 1
    assert array[0, 1] == array[0][1] == 2
    assert array[0, 2] == array[0][2] == 3
    assert array[1, 0] == array[1][0] == 4
    assert array[1, 1] == array[1][1] == 5
    assert array[1, 2] == array[1][2] == 6
    assert array[2, 0] == array[2][0] == 7
    assert array[2, 1] == array[2][1] == 8
    assert array[2, 2] == array[2][2] == 9
```

However, NDArray’s default dtype is float. Please clearly specify `dtype=int` as this can make our circuits more efficient.