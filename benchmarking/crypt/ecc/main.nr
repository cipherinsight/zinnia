fn baby_check(x: Field, y: Field) {
    let x2 = x * x;
    let y2 = y * y;
    let lhs = 168700 * x2 + y2;
    let rhs = 1 + 168696 * x2 * y2;
    assert(lhs == rhs, "Point not on curve");
}

fn baby_add(x1: Field, y1: Field, x2: Field, y2: Field) -> (Field, Field) {
    let beta  = x1 * y2;
    let gamma = y1 * x2;
    let delta = ( -168700 * x1 + y1 ) * ( x2 + y2 );
    let tau   = beta * gamma;

    let denom_x = 1 / ( 1 + 168696 * tau );
    let x = ( beta + gamma ) * denom_x;

    let denom_y = 1 / ( 1 - 168696 * tau );
    let y = ( delta + 168700 * beta - gamma ) * denom_y;

    (x, y)
}

fn main(x1: Field, y1: Field, x2: Field, y2: Field, x3: Field, y3: Field) {
    baby_check(x1, y1);
    baby_check(x2, y2);
    baby_check(x3, y3);

    let (x4, y4) = baby_add(x1, y1, x2, y2);

    assert(x3 == x4, "X-coordinate mismatch");
    assert(y3 == y4, "Y-coordinate mismatch");
}
