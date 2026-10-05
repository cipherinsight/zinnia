# Programming tasks

Implement the requested computation in the prepared project. You may choose your algorithm and use the provided documentation and compatible libraries. Preserve the specified output shape and ordering. Shapes are supplied by the project and may differ between test cases; do not hard-code example answers. Inputs and outputs are integer-valued. The project handles data loading and proof invocation.

All dimensions are positive. Readings, thresholds, coordinates, and image values are integers in `[-512,4096]`; task 6 uses `[0,4096]`. Use signed 64-bit integers in Rust. Project shapes keep every specified intermediate and result within that range of representable integers. Shapes and task 6's round count are known before compilation; array contents vary at execution time.

## 1. Calibrate and clip readings

A sensor's raw readings are stored in `x[N]`. Calibrate each reading by multiplying it by 2 and adding 5, then clamp it to the inclusive interval `[0,255]`.

Return `y[N]`, where `y[i] = min(255, max(0, 2*x[i] + 5))`.

Example: `[-10,0,50,200]` produces `[0,5,105,255]`. Preserve input order. A reading on either clipping boundary remains on that boundary.

## 2. Subtract channel background

The matrix `x[T,C]` contains measurements from `C` sensor channels over `T` samples. For each channel, its background level is the minimum measurement in that column. Subtract the corresponding background level from every measurement.

Return the corrected matrix `y[T,C]`. Both dimensions are positive.

Example:

```text
x = [[3,10],     y = [[0,3],
     [5, 7],          [2,0],
     [4, 9]]          [1,2]]
```

Each column is corrected independently. With only one sample, every corrected measurement is zero.

## 3. Energy above a per-channel noise floor

The matrix `x[N,C]` contains `N` observations of `C` channels. The vector `floor[C]` gives a noise threshold for each channel.

For each observation, sum the squares of the readings that are **strictly greater** than their channel's threshold. Return `energy[N]`. If no reading qualifies, that observation's energy is zero.

Example:

```text
x     = [[2,5,1], [-3,4,8], [0,2,3]]
floor = [0,4,3]
energy = [29,64,0]
```

The selection is per reading, not per row. A reading equal to its threshold is excluded. Input readings and thresholds may be negative.

## 4. Pairwise squared distances

Two collections of points are stored in `x[N,D]` and `y[M,D]`. Compute the squared Euclidean distance between every point in the first collection and every point in the second.

Return `distance[N,M]`, where `distance[i,j]` is the sum, over coordinates `d`, of `(x[i,d]-y[j,d])²`. Do not take a square root.

Example:

```text
x = [[0,0], [1,2]]
y = [[1,0], [3,2]]
distance = [[1,13], [4,4]]
```

Point collections may contain duplicates. Preserve their row order. The numbers of points may differ.

## 5. Sum non-overlapping image blocks

The tensor `x[B,H,W,C]` contains a batch of images with channels in the last dimension. `H` and `W` are positive and even.

For every image and channel, replace each non-overlapping `2×2` block by the sum of its four values. Return `y[B,H/2,W/2,C]`. Keep batches and channels separate.

For one image and one channel:

```text
input plane = [[1,2,3,4],
               [5,6,7,8]]
output plane = [[14,22]]
```

There is no padding and no averaging. The first block covers rows `0,1` and columns `0,1`; the next horizontal block covers columns `2,3`.

## 6. Repeated spatial smoothing

The nonnegative matrix `x[H,W]` is a spatial measurement grid. Apply the following update for the project-specified number of rounds `R`.

At every round, keep all outer-edge cells unchanged. For each interior cell, compute:

```text
next[i,j] = (4*current[i,j]
             + current[i-1,j] + current[i+1,j]
             + current[i,j-1] + current[i,j+1]) // 8
```

Here `//` means integer division rounded down. **Every update in a round uses the grid from the start of that round.** Return the complete grid after `R` rounds.

For one round:

```text
x = [[0,0,0],     result = [[0,0,0],
     [0,8,0],               [0,4,0],
     [0,0,0]]               [0,0,0]]
```

Further rounds continue from the preceding result. A grid with no interior cells remains unchanged. `R` is a small public project parameter, not a data-dependent convergence condition.
