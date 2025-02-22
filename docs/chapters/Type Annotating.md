# Type Annotating

<aside>
🚧

Working in Progress

</aside>

## Input Type Annotating

- `x: int` : declare an input named `x` whose type is `int` . This is a private input by default.
- `x: float` : declare an input named `x` whose type is `float` . This is a private input by default.
- `x: Private[int]` : declare a private input named `x` whose type is `int` .
- `x: Public[int]` : declare a public input named `x` whose type is `int` .
- `x: List[int, float, int]` : declare a private input named `x` whose type is `list`
    - In Zinnia, you should clearly specify the list length and the type of all inner elements
- `x: Tuple[int, int, int]` : declare a private input named `x` whose type is `tuple`
    - In Zinnia, you should clearly specify the tuple length and the type of all inner elements
- `x: NDArray[int, 4, 4]` : declare a private input named `x` whose type is `NDArray` and the shape is `(4, 4)`

## Other Type Annotating

- You are not allowed to use `Private` and `Public` if you are not intended to annotate inputs for a circuit
- You can nest types, e.g. `List[int, List[int, Tuple[float, int]]]`

## Cautions

- Different from Python, you should annotate each element for a list and tuple
- `List[int]` means a list whose length is 1 and the first element is int
- `List[int, float]` means a list whose length is 2 and the first element is int while the second element is float
- This helps us to know the length of a list and elements inside this list at compile time. This helps Zinnia to decide which computation to generate and how many computations.