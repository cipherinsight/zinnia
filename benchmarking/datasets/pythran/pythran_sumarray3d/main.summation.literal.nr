// pythran_sumarray3d__summation: literal translation of kernel.py (LITERAL_BASELINES.md).
// Nested compute is called in comprehension order; each j updates total then x,y,z accumulators. Scalar products round individually.
// @outputs: float[NP]; float[NP * 3]
global NW: u32 = 16; // = weights_0
global NP: u32 = 16; // = points_0
fn compute(
    i: u32,
    pos: [[Field; 3]; NW],
    weights: [Field; NW],
    points: [[Field; 3]; NP],
    sum_array3d: [[Field; 3]; NP],
) -> (Field, [[Field; 3]; NP]) {
    let mut sum_array3d = sum_array3d;
    let pxi = points[i][0];
    let pyi = points[i][1];
    let pzi = points[i][2];
    let mut total: Field = 0;
    for j in 0..NW {
        let weight_j = weights[j];
        let _xj = pos[j][0];
        let _yj = pos[j][1];
        let _zj = pos[j][2];
        let dx = pxi - pos[j][0];
        let dy = pyi - pos[j][1];
        let dz = pzi - pos[j][2];
        let dr = noir_hw::div(
            noir_hw::SCALE,
            noir_hw::sqrt(
                noir_hw::mul(dx, dx) + noir_hw::mul(dy, dy) + noir_hw::mul(dz, dz),
            ),
        );
        total += noir_hw::mul(weight_j, dr);
        sum_array3d[i][0] += noir_hw::mul(weight_j, dx);
        sum_array3d[i][1] += noir_hw::mul(weight_j, dy);
        sum_array3d[i][2] += noir_hw::mul(weight_j, dz);
    }
    (total, sum_array3d)
}

fn main(
    pos: [[Field; 3]; NW],
    weights: [Field; NW],
    points: [[Field; 3]; NP],
) -> pub ([Field; NP], [[Field; 3]; NP]) {
    noir_hw::check_inputs(weights);
    for j in 0..NW {
        noir_hw::check_inputs(pos[j]);
    }
    for i in 0..NP {
        noir_hw::check_inputs(points[i]);
    }
    let mut sum_array3d = [[0; 3]; NP];
    let mut sum_array = [0; NP];
    for i in 0..NP {
        let (total, next) = compute(i, pos, weights, points, sum_array3d);
        sum_array[i] = total;
        sum_array3d = next;
    }
    (sum_array, sum_array3d)
}
