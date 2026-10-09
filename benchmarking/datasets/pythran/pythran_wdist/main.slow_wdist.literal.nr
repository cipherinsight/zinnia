// pythran_wdist__slow_wdist: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each ii,jj builds difference, quotient and square arrays in separate passes, then sum and sqrt.
// @outputs: float[MM * NN]
global K: u32 = 2; // = A_0
global MM: u32 = 3; // = A_1
global NN: u32 = 3; // = B_1

fn main(A: [[Field; MM]; K], B: [[Field; NN]; K], W: [[Field; MM]; K]) -> pub [[Field; NN]; MM] {
    for k in 0..K { noir_hw::check_inputs(A[k]); noir_hw::check_inputs(B[k]); noir_hw::check_inputs(W[k]); }
    let mut d=[[0;NN];MM];
    for ii in 0..MM { for jj in 0..NN {
        let mut diff=[0;K]; for k in 0..K { diff[k]=A[k][ii]-B[k][jj]; }
        let mut wdiff=[0;K]; for k in 0..K { wdiff[k]=noir_hw::div(diff[k],W[k][ii]); }
        let mut squares=[0;K]; for k in 0..K { squares[k]=noir_hw::mul(wdiff[k],wdiff[k]); }
        let mut total: Field=0; for k in 0..K { total+=squares[k]; }
        d[ii][jj]=noir_hw::sqrt(total);
    } }
    d
}
