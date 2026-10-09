// Floyd-Warshall: snapshot each vectorized outer-add/minimum update.
// Compared exact integer sums must fit signed64; reuse selected window proofs.
// @outputs: int[CELLS]
global N: u32 = 16; // = path_0
global COLS: u32 = 16; // = path_1
global CELLS: u32 = N * COLS;
fn main(path: [Field; CELLS]) -> pub [Field; CELLS] {
    let mut path = path;
    let mut bounds = [0 as u64; CELLS];
    for index in 0..CELLS { bounds[index] = noir_hw::window(path[index]); }
    for k in 0..N {
        let mut next = [0; CELLS];
        let mut next_bounds = [0 as u64; CELLS];
        for i in 0..N {
            for j in 0..COLS {
                let index = i * COLS + j;
                let sum = path[i * COLS + k] + path[k * COLS + j];
                let sum_bound = noir_hw::window(sum);
                let take_sum = sum_bound < bounds[index];
                next[index] = if take_sum { sum } else { path[index] };
                next_bounds[index] = if take_sum { sum_bound } else { bounds[index] };
            }
        }
        path = next;
        bounds = next_bounds;
    }
    path
}
