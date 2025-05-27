# Control Flow

Zinnia is a high level language which allows you to write ZK circuits just like writing traditional programs. To provide you with more convenience, we have defined control flows for you to handling branches and loops in Zinnia. You can write control flows by using Python control flow keywords including `if` `else` `elif` `for` `while` and `return` . In general, these control flow is similar with Python while there is minor differences. So let’s kick out this chapter about the control flow in Zinnia!

## Conditional Branching

Conditional Branching is achieved through `if` and `else` . In Zinnia, you can use conditional branching in a way just like vanilla Python. The Zinnia compiler will help you compile the correct constraints which correctly takes care of the conditions for the branches.

### Unknown Conditions

Many ZK compilers do not allow users to write code involving unknown conditions. The “Unknown Conditions” are those branching conditions whose value cannot be known at compile time. Here is an example about the unknown conditions.

```python
@zk_circuit
def condition_unknown(x: int):
    y = 0
    if x == 0:
        y = 1
```

In this example, you may find out that we cannot infer the value to `x` at compile time. This would consequently make the value to the condition of the branch unknown — which is `x == 0` . Many ZK compilers do not allow this, however, Zinnia provides you with this convenience.

### Assignments in Conditional Branching

Python is a weakly typed language — which means that you can declare variables without letting the interpreter know the types to the variables prior to execution. However, in Zinnia, it is crucial for us to know the types to the variables. This information lets Zinnia know which computations and constraints it should generate, and how many of them.

Zinnia ensures the type safety of variables by adding a restriction to condition branches: It is not allowed for users to change the type of variables which are declared outside this scope when the condition is unknown, although you can change the variable of those variables. Please take a look of this example:

```python
@zk_circuit
def foo(cond: int):
    x = [1, 2]
    if cond != 0:
        x = [1, 2, 3]
```

In this example, as we have illustrated before, the branch condition `cond != 0` is unknown at compile time. The variable `x` is declared out of the branching scope, which is declared as a `List[int, int]` type. However, inside the branch scope, the user attempts to change the type of that list into `List[int, int, int]` . This is not allowed as Zinnia compiler is not sure what the type of `x` is after this branching statement.

### Allowed Assignments

However, if we simply reject changing the type of variables outside the branching scope, users would find it very inconvenient for them to write some kinds of code. Take this code as an example:

```python
@zk_circuit
def foo():
    x = []
    for i in range(10):
        if i < 5:
            x.append(i)  # Allowed
```

In this example, the type of variable `x` would be changed each time we append a new integer onto that list. Consider we are append number 1 onto the list. Before the appending operation, the value to `x` is `[0]` and the corresponding type to `x` is `List[int]` . After the append, the value to `x` would be changed to `[0, 1]` and the correspond type to `x` would be changed into `List[int, int]` .

If we don’t allow the list appending in loops and conditional branches, this would make the programming more difficult for users. Instead, we allow the change of datatypes in certain cases, where the condition of branches and loops would be able to be inferred at compile time. In this example, the condition to the append is `i < 5` which can be inferred at compile time so we can safely change the datatype of `x` .

## For-In Loops

In Zinnia, it is allowed to write for-in loops. By using for-in loops, it is nature for us to know the number of loop iterations at compile time, so it is generally safe to do that. 

### Syntax

The syntax of for-in loops is the same with Python. Zinnia compiler will automatically unroll this loop into a known number of iterations. The number of iterations is known because the datatype of `expr` is known at compile time.

```python
for item in expr:
    # some statements or expressions using `item`
```

### Break, Continue and Return

Different from other ZK compilers, Zinnia allows you to take advantage of `break` `continue` and `return` inside loops to manipulate control flows. 

However, there are side effects for using `break` `continue` and `return` inside a loop. If the condition on the `break` `continue` and `return` is unknown at compile time, this would consequently make the condition for the statements in this loop unknown. As a result, the assignment to outer defined variables may be restricted.

## While Loops

In Zinnia, it is allowed to write while loops. The syntax of while loops is same with Python.

### The Condition for While

As we may introduced before, Zinnia need to know the number of iterations of a loop. Thanks to for-in, it allows us to know the number of iterations for a loop without explicitly specify that. However what about while loops?

In Zinnia, our compiler will try our best to infer the maximum number of iterations for a while loop at compile time — though it is not guaranteed for Zinnia to figure out that as it is a Halting Problem which is undecidable in computer science.

In Zinnia, there is a maximum number of iterations configuration. Our compiler will terminate compilation when the while loops reaches that limit. The compiler will throw out a warning or an exception indicating this error.

To conclude, in general, it is not recommended to use `while` loops. Please use for-in loops with a range if possible.