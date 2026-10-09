// npbench_polybench_floyd_warshall_floyd_warshall__kernel:
// literal translation of kernel.py (LITERAL_BASELINES.md).
// Full outer add, full elementwise minimum, then simultaneous slice assignment.
// Compared exact integer sums must fit signed64, as in the optimized circuit.
// @outputs: int[CELLS]
global N: u32 = 16; // = path_0
global COLS: u32 = 16; // = path_1
global CELLS: u32 = N * COLS;
fn main(path: [Field; CELLS]) -> pub [Field; CELLS] {
    noir_hw::check_inputs(path);
    let mut path = path;
    for k in 0..N {
        let mut outer = [0; CELLS];
        for i in 0..N {
            for j in 0..COLS {
                outer[i * COLS + j] = path[i * COLS + k] + path[k * COLS + j];
            }
        }
        let mut minimum = [0; CELLS];
        for index in 0..CELLS {
            minimum[index] = if noir_hw::lt(outer[index], path[index]) {
                outer[index]
            } else { path[index] };
        }
        path = minimum;
    }
    path
}
