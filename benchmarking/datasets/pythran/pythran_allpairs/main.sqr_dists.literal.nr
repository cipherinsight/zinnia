// pythran_allpairs__sqr_dists: literal translation of kernel.py (LITERAL_BASELINES.md).
// Difference and squared arrays are separate full passes; running sum in index order, one rounding per square.
// @outputs: float[NY * NX]
global NX: u32 = 16; // = X_0
global NY: u32 = 16; // = Y_0
global K: u32 = 16; // = X_1

fn main(X: [[Field; K]; NX], Y: [[Field; K]; NY]) -> pub [[Field; NX]; NY] {
    for i in 0..NX { noir_hw::check_inputs(X[i]); } for j in 0..NY { noir_hw::check_inputs(Y[j]); }
    let mut result = [[0; NX]; NY];
    for i in 0..NY { for j in 0..NX {
        let mut diff = [0; K];
        for k in 0..K { diff[k] = X[j][k] - Y[i][k]; }
        let mut squares = [0; K];
        for k in 0..K { squares[k] = noir_hw::mul(diff[k], diff[k]); }
        let mut sum: Field = 0;
        for k in 0..K { sum += squares[k]; }
        result[i][j] = sum;
    } }
    result
}
